#
# pproc_worker.py - DeGirum Python SDK: postprocessor workers factory
# Copyright DeGirum Corp. 2024
#
# Provides create_worker function to start workers in separate processes
#

import sys
import logging

if "darwin" in sys.platform:
    import site

    sys.path.append(site.getusersitepackages())

import importlib
import json

from queue import Queue
from threading import Thread, Event, Lock
from multiprocessing import set_start_method, set_executable

import argparse
import os
import tempfile, shutil
from pathlib import Path
import signal
import time

import subprocess
import psutil

import numpy as np

import msgpack
import msgpack_numpy
import zmq

msgpack_numpy.patch()

if "linux" in sys.platform:
    import pyseccomp as seccomp
    import resource as rc

    MEMORY_LIMIT = 1024  # Mb
else:
    MEMORY_LIMIT = -1  # not set

import hashlib, uuid
import pickle

# modules allowed to import in postprocessors
allowed_modules = ["numpy", "cv2"]


def create_worker(
    parent_pid: int = 0,
    port_id: int = 5555,
    protocol: str = "tcp",
    ipc_dir: str = "tmp",
    memory_limit_MB: int = MEMORY_LIMIT,
    time_limit_sec: float = 1.0,
    log_level=logging.INFO,
):
    """
    Create a worker for postprocess execution
    parent_pid: int: parent process id; 0 if there is none
    port_id: int: port to bind the 0mq socket
    protocol: str: data transfer protocol to use
    ipc_dir: str: temp folder for ipc protocol
    memory_limit_MB: int: process address space limit, in Mb
    time_limit_sec: float: process execution time limit, in sec
    """
    set_start_method("spawn", force=True)

    command = [
        get_python(),
        os.path.abspath(__file__),
        "--parent_pid",
        str(parent_pid),
        "--port_id",
        str(port_id),
        "--protocol",
        protocol,
        "--ipc_dir",
        ipc_dir,
        "--memory_limit_MB",
        str(memory_limit_MB),
        "--time_limit_sec",
        str(time_limit_sec),
    ]

    # Start the process
    worker = subprocess.Popen(command, stderr=subprocess.PIPE)

    status = -1
    for i in range(100):
        try:
            if worker.stderr is not None:
                res: dict = json.loads(worker.stderr.readline().rstrip())
                status = int(res[PP_Worker.data_key])
                # got process status:
                # 0: socket binding OK; 1: failed to bind socket, exiting
                break
        except Exception:
            time.sleep(0.01)

    if status == -1:
        # unexpected behaviour
        worker.kill()
        return None, status

    if status == 1 or status == 2:
        # subprocess is exiting
        worker.wait(1.0)
        return None, status

    return worker, 0


def is_worker_running(worker: subprocess.Popen) -> bool:
    """
    Checks if a subprocess is running
    Returns 1 if yes, 0 otherwise
    """
    return worker.poll() is None


def stop_worker(
    worker: subprocess.Popen, port_id: int, protocol: str, ipc_dir: str
) -> int:
    """
    Stop a worker.
    Input: worker: a subprocess worker to be stoped;
    port_id: what 0mq port the worker is using;
    protocol: wwhat protocol the worker is using (tcp or ipc);
    ipc_dir: what temp folder for ipc protocol the worker is using
    Returns 0 on success; 1 if process is not running in the first place
    """
    if worker.poll() is not None:
        return 1  # subprocess is not running

    context = zmq.Context()
    try:
        socket = context.socket(zmq.REQ)
        if protocol == "tcp":
            socket.connect(f"tcp://localhost:{port_id}")
        elif protocol == "ipc":
            socket.connect(f"ipc://{ipc_dir}")

        req = {"action": "poison_pill"}
        socket.send(msgpack.packb(req))
        reply = msgpack.unpackb(socket.recv())
        assert reply[PP_Worker.ret_val_key].lower() == "ok"
        worker.wait(1.0)
    except Exception:
        worker.kill()
        worker.wait(1.0)

    context.destroy()
    return 0


def tostr(x):
    if isinstance(x, bytes):
        return x.decode()
    else:
        return str(x)


def _unpack_dtype(dtype):
    """
    Unpack dtype descr, recursively unpacking nested structured dtypes.
    """

    if isinstance(dtype, (list, tuple)):
        # Unpack structured dtypes of the form: (name, type, *shape)
        dtype = [
            (subdtype[0], _unpack_dtype(subdtype[1])) + tuple(subdtype[2:])
            for subdtype in dtype
        ]
    return np.dtype(dtype)


def extract_exc_info(e):
    """
    Extracts exception information: file, function, line number and exception message.
    """
    tb = e.__traceback__
    while tb.tb_next:  # Traverse to the last frame
        tb = tb.tb_next
    frame = tb.tb_frame
    filename = os.path.basename(frame.f_code.co_filename)
    return f"{e} [{type(e).__name__}] in file '{filename}', function '{frame.f_code.co_name}', line {tb.tb_lineno}"


def to_ndarray(obj):
    """
    Decoder for deserializing numpy data types.
    """
    try:
        if "nd" in obj:
            if obj["nd"] is True:
                # Check if b'kind' is in obj to enable decoding of data
                # serialized with older versions (#20) or data
                # that had dtype == 'O' (#46):
                if "kind" in obj and obj["kind"] == "V":
                    descr = [
                        tuple(tostr(t) if type(t) is bytes else t for t in d)
                        for d in obj["type"]
                    ]
                elif "kind" in obj and obj["kind"] == b"O":
                    return pickle.loads(obj["data"])
                else:
                    descr = obj["type"]

                dtype_ = _unpack_dtype(descr)
                arr: np.ndarray = np.ndarray(
                    buffer=obj["data"], dtype=dtype_, shape=obj["shape"]
                )
                return arr
            else:
                descr = obj["type"]
                dtype_ = _unpack_dtype(descr)
                return np.frombuffer(obj["data"], dtype=dtype_)[0]
        elif "complex" in obj:
            return complex(tostr(obj[b"data"]))
        else:
            raise KeyError("no known key")
    except KeyError:
        return None


def create_module_key(fpath: Path) -> str:
    """
    create a unique id for a file
    Input:
    fpath: Path: file path
    Output:
    unique id for thr given file
    """
    hash = ""
    with open(fpath, "rb") as f:
        data = f.read()
        hash = hashlib.md5(data).hexdigest()
    return hash


def import_postprocessor(mod_path_str: str):
    mod_path: Path = Path(mod_path_str)
    mod_key: str = create_module_key(mod_path)
    mod_name: str = f"{mod_path.stem}_{str(mod_key)}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir, mod_name).with_suffix(mod_path.suffix)

    shutil.copy(str(mod_path), str(temp_file))
    sys.path.append(temp_dir)
    module = importlib.import_module(mod_name)

    sys.path.remove(temp_dir)
    shutil.rmtree(temp_dir)

    return module


class PP_Worker:
    """
    Store postpocess classes and execute them when requested.
    In case of runaway execution, monitor thread will terminate the process
    """

    poison_pill: str = "poison_pill"
    time_limit: float = 1.0  # limit postproc execution by 1 sec
    socket_timeout_sec: float = 1.0  # socket timeout

    # Return json keys
    ret_val_key: str = "rv"
    msg_key: str = "msg"
    data_key: str = "data"
    err_code_key: str = "err_code"

    # Error codes
    err_code_other = -1
    err_code_terminated = 1
    err_code_postprocessor_missing = 2
    err_code_postprocessor_unavailable = 3
    err_code_postprocessor_semantics = 4

    logger = logging.getLogger(__name__)

    @classmethod
    def set_time_limit(cls, time_limit: float = 1.0):
        cls.time_limit = time_limit

    @classmethod
    def set_log_level(cls, log_level):
        cls.logger.setLevel(log_level)

    def __init__(
        self, parent_pid: int, port_id: int, protocol: str = "tcp", ipc_dir: str = "tmp"
    ):
        """
        port_id: int: port to bind the 0mq socket
        protocol: str: data transfer protocol to use
        ipc_dir: str : ipc data folder
        """
        protocols_supported = ["tcp", "ipc"] if "linux" in sys.platform else ["tcp"]
        if protocol not in protocols_supported:
            raise ValueError(f"PP_Worker: {protocol} not supported on {sys.platform}")

        self.parent_pid = parent_pid
        self.cache: dict = {}  # loaded postprocs

        # initialize 0mq
        self.binded_addr = ""
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, int(1000 * PP_Worker.socket_timeout_sec))
        try:
            if protocol == "tcp":
                self.socket.bind(f"{protocol}://*:{port_id}")
                self.binded_addr = f"{protocol}://*:{port_id}"
            elif protocol == "ipc":
                self.socket.bind(f"{protocol}://*:{ipc_dir}")
                self.binded_addr = f"{protocol}://*:{port_id}"
        except Exception:
            self.release()

        self.stop_event = Event()  # set when process is being terminated by monitor
        self.send_lock = Lock()  # guards responce sending

        self.monitor_thread: Thread = Thread()  # monitoring thread; starts in run
        self.monitor_q: Queue = Queue()  # pass postprocessor id to monitor thread
        self.monitor_timeout = Event()  # terminate the worker's runaway process

    def release(self):
        """
        Release resources.
        """
        self.zmq_context.destroy()

    def _add_pp(self, data: dict):
        """
        Add a postpocessor, if it has a new id. Increase ref count, if it exists already.
        data : dict: should contains path and config
        """
        pp_id = str(uuid.uuid4())
        if pp_id is not None:
            self.monitor_q.put_nowait(pp_id)

        pp_path = data["pp_path"]
        json_config = data["json_config"]

        PP_Worker.logger.info(f"Adding postproc {pp_path}")

        if not os.path.isfile(pp_path):
            return PP_Worker.err_reply(
                f"{pp_path} is not available",
                err_code=PP_Worker.err_code_postprocessor_unavailable,
            )
        pp_module = import_postprocessor(pp_path)
        PP_Worker.logger.debug(f"imported postproc {pp_path}")

        if hasattr(pp_module.PostProcessor, "configure"):
            pp = pp_module.PostProcessor()
            PP_Worker.logger.debug("calling configure")
            pp.configure(json_config)
        else:
            if not isinstance(json_config, str):
                if isinstance(json_config, dict):
                    json_config_str = json.dumps(json_config)
                else:
                    json_config_str = str(json_config)
            PP_Worker.logger.debug("calling constructor")
            pp = pp_module.PostProcessor(json_config_str)

        self.cache[pp_id] = pp
        PP_Worker.logger.info(f"add_pp: pp_id: {pp_id}, {id(pp)}")
        return PP_Worker.ok_reply({"pp_id": pp_id})

    def _forward(self, data: dict):
        """
        Pass data to postprocessor's 'forward' function.
        data : dict: should contains target postprocessor id, tensor_list and details_list
        """
        # unpack the needed data
        pp_id = data["pp_id"]
        PP_Worker.logger.info(f"Forward: pp_id: {pp_id}")
        if pp_id not in self.cache.keys():
            PP_Worker.logger.info("Forward: postproc missing")
            return PP_Worker.err_reply(
                "PP_Worker._forward: Postprocessor is missing.",
                err_code=PP_Worker.err_code_postprocessor_missing,
            )

        basic_tensor_list = data.setdefault("tensor_list", [])
        details_list = data.setdefault("details_list", [])

        # if needed, convert basic tensors (decoded as dicts) to numpy arrays
        tensor_list = []
        if basic_tensor_list is not None:
            for t in basic_tensor_list:
                if isinstance(t, dict):
                    tensor_list.append(to_ndarray(t))
                else:
                    tensor_list.append(t)

        # get the requested postprocessor and call forward
        pp = self.cache[pp_id]
        PP_Worker.logger.info(f"Forward: calling forward: {id(pp)}, {self.cache}")
        reply = pp.forward(tensor_list, details_list)

        if isinstance(reply, str):
            # that's old-style postprocessor result, convert it to json
            reply_json = json.loads(reply)
        else:
            reply_json = reply

        return PP_Worker.ok_reply(reply_json)

    def _poison_pill(self, data: dict):
        """
        Process the poison pill.
        data : dict: unused
        """
        PP_Worker.logger.info("Got poison pill!")
        return PP_Worker.ok_reply("got poison pill")

    def _release_pp(self, data: dict):
        """
        Remove the postprocessor with the given id.
        data : dict: should contain id of the postprocessor to be deleted
        """
        pp_id = data["pp_id"]
        if pp_id in self.cache:
            del self.cache[pp_id]
            return PP_Worker.ok_reply({"pp_id": pp_id})
        else:
            return PP_Worker.err_reply(f"postprocessor {pp_id} is not loaded")

    def _get_info(self, data: dict):
        """
        Get worker info: a stub; more diagnostics will be added
        data : dict: unused
        """
        return PP_Worker.ok_reply(
            {
                "pproc_num": len(self.cache),
                "time_limit": PP_Worker.time_limit,
                "binded_addr": self.binded_addr,
            }
        )

    def check_parent(self):
        """
        Check For the existence of a unix pid.
        """
        if self.parent_pid == 0 or psutil.pid_exists(self.parent_pid):
            return True
        else:
            return False

    def run(self):
        """
        Server request/reply loop.
        """
        PP_Worker.logger.debug("---------- Entered PP_Worker.run! ---------------")
        PP_Worker.logger.debug(f"PP_Worker.run : timeout: {PP_Worker.time_limit}")

        self.monitor_thread = Thread(target=self.monitoring)
        self.monitor_thread.start()

        action: str = ""
        while action != PP_Worker.poison_pill:
            # wait for request
            PP_Worker.logger.debug("PP_Worker.run: wait for request")
            try:
                request = self.socket.recv()
            except Exception:
                if not self.check_parent():
                    break

            try:
                # convert to json
                data: dict = msgpack.unpackb(request)
                # extract action
                action = data["action"]
                PP_Worker.logger.debug(f"PP_Worker.run: action: {action}")
                pp_id = data.setdefault("pp_id", None)
                # NB: by adding like 'if action == "forward"' clause, you  may impose additional conditions
                # on what type of actions' execution is monitored.
                if pp_id is not None:
                    self.monitor_q.put_nowait(pp_id)
                # process requested self._action; self.data_err will be called if self._action is missing
                reply: bytes = getattr(self, "_" + action, self.data_err)(data)

            except Exception as e:
                msg = extract_exc_info(e)
                PP_Worker.logger.info("PP_Worker.run: {msg}")
                reply = PP_Worker.err_reply(msg)

            finally:
                # cancel monitor timer to avoid terminating by timeout
                self.monitor_q.join()
                self.monitor_timeout.set()
            # send reply
            if not self.stop_event.is_set():
                self.send_reply(reply)

        # stop monitor
        self.monitor_q.put_nowait(None)
        self.monitor_thread.join()

        # release resouces
        self.release()
        PP_Worker.logger.info("PP_Worker.run: Worker exits")

    def monitoring(self):
        """
        Monitoring thread function
        """
        while True:
            # wait indefinetely for a postprocessor to be run
            pp_id = self.monitor_q.get()
            if pp_id is None:
                # got the poison pill
                self.monitor_q.task_done()
                break
            # start timer
            self.monitor_timeout.clear()
            self.monitor_q.task_done()
            if not self.monitor_timeout.wait(PP_Worker.time_limit):
                self.terminate("Max running time exceeded")
            else:
                pass

    def terminate(self, reason: str = ""):
        """
        Send the 'reason' as the reply, release resources, then terminate the current process.
        """
        self.stop_event.set()
        if len(reason) > 0:
            reply = self.err_reply(
                f"Postprocessor terminated: {reason}", PP_Worker.err_code_terminated
            )
            self.send_reply(reply)
        self.release()
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)

    def send_reply(self, reply: bytes) -> bool:
        """
        Send reply via 0mq socket.
        reply: bytes: data to send
        Return: bool: True on success, False otherwise
        """
        with self.send_lock:
            try:
                self.socket.send(reply)
                return True
            finally:
                return False

    @classmethod
    def data_err(cls, data: dict) -> bytes:
        return PP_Worker.err_reply("wrong request format")

    @classmethod
    def err_reply(cls, err_msg: str, err_code=err_code_other) -> bytes:
        return msgpack.packb(
            {
                PP_Worker.ret_val_key: "Err",
                PP_Worker.msg_key: err_msg,
                PP_Worker.err_code_key: err_code,
            }
        )

    @classmethod
    def ok_reply(cls, data) -> bytes:
        res = {PP_Worker.ret_val_key: "OK", PP_Worker.data_key: data}
        return msgpack.packb(res)


def get_python():
    """
    Returns the path of the Python interpreter to use when starting a child process.
    """
    if "win32" in sys.platform:
        py_exec_path = os.path.join(sys.exec_prefix, "python.exe")
    elif "linux" in sys.platform:
        py_exec_path = sys.exec_prefix + "/bin/python3"
    elif "darwin" in sys.platform:
        py_exec_path = sys.exec_prefix + "/bin/python3"
    else:
        py_exec_path = ""

    return py_exec_path


def configure_embed():
    """
    1. Sets the path of the Python interpreter to use when starting a child process.
    2. Set the method which should be used to start child processes as 'spawn'.
    """
    py_exec_path = get_python()

    if os.path.isfile(py_exec_path):
        set_executable(py_exec_path)
    else:
        PP_Worker.logger.error(f"cannot set_executable: {py_exec_path} is not a file")

    # set_start_method("spawn", force=True)
    try:
        set_start_method("spawn", force=False)
    except Exception as e:
        PP_Worker.logger.debug(f"set_start_method exception: {e}")


def set_mem_limit(memory_limit_bytes: int = -1):
    """
    Set memory usage limit
    memory_limit: int: process address space limit, in bytes
    """
    # set max size of Process Address Space
    if "linux" in sys.platform and memory_limit_bytes > 0:
        # load all allowed modules
        for module in allowed_modules:
            importlib.import_module(module)

        # find used heap size
        current_process = psutil.Process(os.getpid())
        memory_info = current_process.memory_info()
        # add user-defined cap
        allowed_memory_bytes = memory_info.data + memory_limit_bytes
        # set DATA limit
        rc.setrlimit(rc.RLIMIT_DATA, (allowed_memory_bytes, allowed_memory_bytes))  # type: ignore


def secure():
    if "linux" not in sys.platform:
        return
    syscall2deny = [
        "execve",
        "execveat",
        "fork",
        "vfork",
        "connect",
    ]
    # Create a seccomp filter with default action ALLOW
    filter = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    action_deny = seccomp.ERRNO(seccomp.errno.EPERM)
    # Deny specific syscalls
    for syscall in syscall2deny:
        filter.add_rule(action_deny, syscall)
    # Load the filter
    filter.load()


def run_worker(
    parent_pid: int = 0,
    port_id: int = 5555,
    protocol: str = "tcp",
    ipc_dir: str = "tmp",
    memory_limit_MB: int = MEMORY_LIMIT,
    time_limit_sec: float = 1.0,
    log_level=logging.INFO,
):
    """
    port_id: int: port to bind the 0mq socket
    protocol: str: data transfer protocol to use
    ipc_dir: str: temp folder for ipc protocol
    memory_limit_MB: int: process address space limit, in Mb
    time_limit_sec: float: process execution time limit, in sec
    """
    if memory_limit_MB > 0:
        memory_limit_bytes = memory_limit_MB * 1024 * 1024
        set_mem_limit(memory_limit_bytes)

    secure()

    PP_Worker.set_time_limit(time_limit_sec)
    PP_Worker.set_log_level(log_level)

    try:
        worker = PP_Worker(
            parent_pid=parent_pid, port_id=port_id, protocol=protocol, ipc_dir=ipc_dir
        )
        if worker.binded_addr != "":
            # binding success, report and start running
            ret_msg_0: str = json.dumps(
                {PP_Worker.ret_val_key: "OK", PP_Worker.data_key: 0}
            )
            sys.stderr.write(ret_msg_0 + "\n")
            sys.stderr.flush()
            worker.run()
        else:
            # binding failed, report and exit process
            ret_msg_1: str = json.dumps(
                {PP_Worker.ret_val_key: "OK", PP_Worker.data_key: 1}
            )
            sys.stderr.write(ret_msg_1 + "\n")
            sys.stderr.flush()
    except Exception as e:
        PP_Worker.logger.error(f"run_worker: PP_Worker constructor throws: {e}")
        ret_msg_2: str = json.dumps(
            {PP_Worker.ret_val_key: "OK", PP_Worker.data_key: 2}
        )
        sys.stderr.write(ret_msg_2 + "\n")
        sys.stderr.flush()


def start_worker():
    """
    Parse arguments and run worker
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--parent_pid", help="parent process pid; 0 if there is none", default="0"
    )
    parser.add_argument(
        "-p", "--port_id", help="port to be used for ZeroMQ socket", default="5000"
    )
    parser.add_argument(
        "-r", "--protocol", help="protocol used by ZeroMQ socket", default="tcp"
    )
    parser.add_argument(
        "-i",
        "--ipc_dir",
        help="directory to be used by ZeroMQ socket for ipc communication",
        default="tmp",
    )
    parser.add_argument(
        "-m",
        "--memory_limit_MB",
        help="process address space limit, in Mb",
        default=str(MEMORY_LIMIT),
    )
    parser.add_argument(
        "-t",
        "--time_limit_sec",
        help=" process execution time limit, in sec",
        default=str(1.0),
    )

    try:
        args = parser.parse_args()
        parent_pid: int = int(args.parent_pid)
        port_id: int = int(args.port_id)
        protocol: str = args.protocol
        ipc_dir: str = args.ipc_dir
        memory_limit_MB: int = int(args.memory_limit_MB)
        time_limit_sec: float = float(args.time_limit_sec)
    except Exception as e:
        raise Exception("start_worker: " + str(e))

    log_level = logging.INFO

    run_worker(
        parent_pid,
        port_id,
        protocol,
        ipc_dir,
        memory_limit_MB,
        time_limit_sec,
        log_level,
    )


if __name__ == "__main__":
    start_worker()
