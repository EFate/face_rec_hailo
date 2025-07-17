# ui.py

import streamlit as st
import requests
import pandas as pd
from typing import Tuple, Any, Dict, List
from pathlib import Path
import time
import json
from datetime import datetime
import os

# ==============================================================================
# 1. 页面配置与美化 (Page Config & Styling)
# ==============================================================================

st.set_page_config(
    page_title="人脸识别系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 全新设计的CSS样式 ---
st.markdown("""
<style>
    /* --- 全局与字体 --- */
    .stApp { background-color: #f0f2f6; }
    h1, h2, h3 { font-weight: 700; color: #1a1f36; }

    /* --- 侧边栏 --- */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e4e8;
    }

    /* --- 使用 st.radio 模拟 Tabs 的核心CSS --- */
    [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 1.5rem;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0;
        margin-bottom: 1.5rem;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] > label {
        height: 50px;
        padding: 0 1rem;
        background-color: transparent;
        border-bottom: 4px solid transparent;
        border-radius: 0;
        font-weight: 600;
        color: #6c757d;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        margin: 0;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        border-bottom: 4px solid #4f46e5;
        color: #4f46e5;
    }

    /* --- 自定义信息卡片 --- */
    .info-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #e0e4e8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease;
        height: 100%; /* 让卡片等高 */
    }
    .info-card:hover { transform: translateY(-5px); }
    .info-card .icon { font-size: 2.5rem; }
    .info-card .title { font-weight: 600; color: #6c757d; font-size: 1rem; margin-top: 10px; }
    .info-card .value { font-weight: 700; color: #1a1f36; font-size: 2rem; }

    /* --- 其他美化 --- */
    .stButton>button { border-radius: 8px; font-weight: 600; }
    [data-testid="stExpander"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. 会话状态管理 (Session State Management)
# ==============================================================================

def initialize_session_state():
    """初始化应用所需的全部会话状态。"""
    backend_host = os.getenv("UI_HOST", "localhost")
    defaults = {
        "api_url": f"{backend_host}:12010",  # 默认指向后端的12010端口
        "api_status": (False, "尚未连接"),
        "faces_data": None,
        "show_register_dialog": False,
        "active_page": "仪表盘",
        "viewing_stream_info": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==============================================================================
# 3. API通信与数据处理 (API Communication & Data Handling)
# ==============================================================================

API_ENDPOINTS = {
    'HEALTH': '/api/face/health',
    'FACES': '/api/face/faces',
    'FACE_BY_SN': '/api/face/faces/{}',
    'RECOGNIZE': '/api/face/recognize',
    'STREAMS_START': '/api/face/streams/start',
    'STREAMS_STOP': '/api/face/streams/stop/{}',
    'STREAMS_LIST': '/api/face/streams',
}


@st.cache_data(ttl=10)
def check_api_status(api_url: str) -> Tuple[bool, str]:
    """检查后端API的健康状况。"""
    try:
        url = f"http://{api_url}{API_ENDPOINTS['HEALTH']}"
        response = requests.get(url, timeout=3)
        if response.ok:
            return True, response.json().get("data", {}).get("message", "服务运行正常")
        return False, f"服务异常 (HTTP: {response.status_code})"
    except requests.RequestException:
        return False, "服务连接失败"


def parse_error_message(response: requests.Response) -> str:
    """智能解析后端的错误信息。"""
    try:
        res_json = response.json()
        # FastAPI 验证错误的标准格式
        if "detail" in res_json:
            detail = res_json["detail"]
            if isinstance(detail, list) and detail:
                first_error = detail[0]
                # 从 loc 数组中提取字段名
                field_location = " → ".join(map(str, first_error.get("loc", [])))
                message = first_error.get("msg", "未知验证错误")
                # 美化字段位置的显示
                field_location = field_location.replace("body", "请求体").replace("query", "查询参数").replace("form",
                                                                                                               "表单")
                return f"字段 '{field_location}' 无效: {message}"
            elif isinstance(detail, str):
                return detail
        # 自定义 ApiResponse 的错误格式
        if "msg" in res_json and res_json.get("code") != 0:
            return res_json["msg"]
        return response.text
    except json.JSONDecodeError:
        return f"无法解析响应 (HTTP {response.status_code})"


def api_request(method: str, endpoint: str, **kwargs) -> Tuple[bool, Any, str]:
    """统一的API请求函数。"""
    full_url = f"http://{st.session_state.api_url}{endpoint}"
    try:
        response = requests.request(method, full_url, timeout=10, **kwargs)
        if response.ok:
            # 对于 204 No Content 这样的成功响应，没有 body
            if response.status_code == 204 or not response.content:
                return True, None, "操作成功"
            res_json = response.json()
            # 检查我们自定义的 ApiResponse 结构
            if res_json.get("code") == 0:
                return True, res_json.get("data"), res_json.get("msg", "操作成功")
            else:
                return False, None, res_json.get("msg", "发生未知错误")
        else:
            error_message = parse_error_message(response)
            return False, None, error_message
    except requests.RequestException as e:
        return False, None, f"网络请求失败，请检查后端地址或服务状态: {e}"


def refresh_all_data():
    """从API获取最新的人脸库数据。"""
    st.session_state.faces_data = {"count": 0, "faces": [], "unique_sns": []}  # 先清空
    with st.spinner("正在从服务器同步最新数据..."):
        success, data, msg = api_request('GET', API_ENDPOINTS['FACES'])
        if success and data:
            all_faces = data.get('faces', [])
            unique_sns = sorted(list({face['sn'] for face in all_faces}))
            st.session_state.faces_data = {
                "count": data.get('count', 0),
                "faces": all_faces,
                "unique_sns": unique_sns
            }
            st.toast("人脸库数据已同步!", icon="🔄")
        else:
            st.error(f"人脸库数据加载失败: {msg}")


def convert_path_to_url(server_path: str) -> str:
    """将后端返回的文件路径智能地转换为可访问的URL。"""
    if not server_path or not isinstance(server_path, str):
        return "https://via.placeholder.com/150?text=No+Path"
    # 使用 as_posix() 确保是 / 分隔符
    p = Path(server_path).as_posix()
    if 'data/' in p:
        # 稳健地分割路径
        relative_path = p.split('data/', 1)[1]
        return f"http://{st.session_state.api_url}/data/{relative_path}"
    return f"https://via.placeholder.com/150?text=Path+Error"


def format_datetime_human(dt_str: str) -> str:
    """将ISO格式的日期时间字符串转换为人性化的格式"""
    if not dt_str:
        return "永久"
    try:
        dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "N/A"


# ==============================================================================
# 4. UI渲染模块 (UI Rendering Modules)
# ==============================================================================

def render_sidebar():
    """渲染侧边栏。"""
    with st.sidebar:
        st.title("🤖 人脸识别系统")
        st.caption("v1.0")

        st.session_state.api_url = st.text_input("后端服务地址", value=st.session_state.api_url,
                                                 help="例如: 192.168.1.15:12010, 请指定正确的服务ip;")

        is_connected, status_msg = check_api_status(st.session_state.api_url)
        st.session_state.api_status = (is_connected, status_msg)
        status_icon = "✅" if is_connected else "❌"
        st.info(f"**API状态:** {status_msg}", icon=status_icon)

        st.divider()
        if st.button("🔄 强制刷新全站数据", use_container_width=True):
            refresh_all_data()

        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
        st.info("人脸识别系统 v1.0")


@st.dialog("➕ 注册新人员", width="large")
def render_register_dialog():
    """渲染用于注册新人员的弹窗。"""
    st.subheader("新人员信息录入")
    with st.form("new_person_form"):
        col1, col2 = st.columns(2)
        name = col1.text_input("姓名", placeholder="例如：张三")
        sn = col2.text_input("唯一编号(SN)", placeholder="例如：EMP001")
        image_file = st.file_uploader("上传人脸照片", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

        if st.form_submit_button("✔️ 确认注册", type="primary", use_container_width=True):
            if name and sn and image_file:
                with st.spinner("正在注册新人员..."):
                    # 将 name 和 sn 作为表单数据 (data) 发送，文件作为 files 发送
                    form_data = {'name': name, 'sn': sn}
                    files_payload = {'image_file': (image_file.name, image_file.getvalue(), image_file.type)}

                    success, data, msg = api_request(
                        'POST',
                        API_ENDPOINTS['FACES'],
                        data=form_data,
                        files=files_payload
                    )
                if success:
                    st.toast(f"注册成功!", icon="🎉")
                    refresh_all_data()
                    st.session_state.show_register_dialog = False
                    st.rerun()
                else:
                    st.error(f"注册失败: {msg}")
            else:
                st.warning("姓名、SN和照片均为必填项。")

    if st.button("取消", use_container_width=True):
        st.session_state.show_register_dialog = False
        st.rerun()


def render_dashboard_page():
    """渲染仪表盘页面。"""
    st.header("📊 仪表盘总览")

    is_connected, _ = st.session_state.api_status
    if not is_connected:
        st.warning("API服务未连接，请在左侧侧边栏配置正确的服务地址并确保后端服务已启动。页面功能将受限。")
        return

    # 刷新按钮
    if st.button("刷新统计信息", type="primary"):
        refresh_all_data()
        # 清除缓存以重新获取视频流信息
        st.cache_data.clear()

    # 确保 faces_data 不是 None，如果 st.session_state.get 返回 None，则使用空字典
    faces_data = st.session_state.get("faces_data") or {}
    unique_sns_count = len(faces_data.get('unique_sns', []))
    api_status, api_color = ("在线", "#28a745") if st.session_state.api_status[0] else ("离线", "#dc3545")

    # 获取视频流数量
    success, stream_data, _ = api_request("GET", API_ENDPOINTS['STREAMS_LIST'])
    stream_count = stream_data.get('active_streams_count', 0) if success else "N/A"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.html(f"""
        <div class="info-card">
            <div class="icon">👥</div>
            <div class="title">人脸库人员总数</div>
            <div class="value">{unique_sns_count}</div>
        </div>""")
    with col2:
        st.html(f"""
        <div class="info-card">
            <div class="icon">📡</div>
            <div class="title">API服务状态</div>
            <div class="value" style="color:{api_color};">{api_status}</div>
        </div>""")
    with col3:
        st.html(f"""
        <div class="info-card">
            <div class="icon">📹</div>
            <div class="title">当前活动视频流</div>
            <div class="value">{stream_count}</div>
        </div>""")

    st.divider()
    st.header("🧐 快速人脸识别")
    with st.container(border=True):
        uploaded_file = st.file_uploader("上传图片进行识别", type=["jpg", "png", "jpeg"], key="recognize_uploader")
        if uploaded_file:
            col_img, col_res = st.columns([0.6, 0.4])
            with col_img:
                st.image(uploaded_file, caption="待识别图片预览", use_container_width=True)
            with col_res:
                st.subheader("识别结果")
                with st.spinner("正在识别中..."):
                    files = {'image_file': (uploaded_file.name, uploaded_file.getvalue())}
                    success, data, msg = api_request('POST', API_ENDPOINTS['RECOGNIZE'], files=files)
                if success:
                    if data:
                        st.success(f"识别成功！找到 {len(data)} 个匹配项。")
                        for result in data:
                            with st.container(border=True):
                                st.markdown(f"**姓名:** {result.get('name')} | **SN:** {result.get('sn')}")
                                similarity_percent = (1 - result.get('distance', 1)) * 100
                                st.markdown(
                                    f"**置信度:** <span style='color:green; font-weight:bold;'>{similarity_percent:.2f}%</span> (距离: {result.get('distance', 0):.4f})",
                                    unsafe_allow_html=True)
                    else:
                        st.info("图像中检测到人脸，但未在库中找到匹配项。")
                else:
                    st.error(f"识别失败: {msg}")


def render_management_page():
    """渲染人脸库管理页面。"""
    st.header("🗂️ 人脸库管理中心")
    if st.button("➕ 注册新人员", type="primary"):
        st.session_state.show_register_dialog = True

    if st.session_state.get("show_register_dialog"):
        render_register_dialog()

    st.divider()

    # 确保 faces_data 不是 None，如果 st.session_state.get 返回 None，则使用空字典
    faces_data = st.session_state.get("faces_data") or {}
    if not faces_data.get('unique_sns'):
        st.info("人脸库为空，或数据加载中... 请确保API服务在线并尝试刷新数据。")
        return

    unique_sns = faces_data.get('unique_sns', [])
    all_faces_info = faces_data.get('faces', [])
    st.subheader(f"👥 人员列表 (共 {len(unique_sns)} 人)")

    num_cols = 3
    cols = st.columns(num_cols)
    for i, sn in enumerate(unique_sns):
        col = cols[i % num_cols]
        person_faces = [f for f in all_faces_info if f['sn'] == sn]
        if not person_faces: continue
        name = person_faces[0]['name']

        with col, st.container(border=True):
            st.markdown(f"#### {name}")
            st.caption(f"SN: {sn}")

            # 使用更紧凑的图像展示
            img_captions = [f"ID: ...{face['uuid'][-4:]}" for face in person_faces]
            img_urls = [convert_path_to_url(face.get('image_path')) for face in person_faces]
            st.image(img_urls, width=60, caption=img_captions)

            with st.expander("⚙️ 管理此人"):
                # 更新姓名
                new_name = st.text_input("更新姓名", value=name, key=f"update_name_{sn}")
                if st.button("✔️ 确认更新", key=f"update_btn_{sn}", use_container_width=True):
                    if new_name and new_name != name:
                        with st.spinner("正在更新..."):
                            endpoint = API_ENDPOINTS['FACE_BY_SN'].format(sn)
                            success, _, msg = api_request('PUT', endpoint, json={"name": new_name})
                        if success:
                            st.toast(f"'{name}' 已更新为 '{new_name}'", icon="✅")
                            refresh_all_data()
                            st.rerun()
                        else:
                            st.error(f"更新失败: {msg}")

                st.divider()

                # 删除人员
                if st.button("🗑️ 删除此人所有记录", key=f"delete_{sn}", use_container_width=True, type="secondary"):
                    with st.spinner("正在删除..."):
                        endpoint = API_ENDPOINTS['FACE_BY_SN'].format(sn)
                        success, _, msg = api_request('DELETE', endpoint)
                    if success:
                        st.toast(f"'{name}' ({sn}) 已被删除。", icon="🗑️")
                        refresh_all_data()
                        st.rerun()
                    else:
                        st.error(f"删除失败: {msg}")


def render_monitoring_page():
    """渲染实时视频监控页面。"""
    st.header("🛰️ 实时视频监测")

    with st.expander("▶️ 启动新监测任务", expanded=True):
        with st.form("start_stream_form"):
            source = st.text_input("视频源", "0", help="可以是摄像头ID(如 0, 1) 或 视频文件/URL")
            lifetime = st.number_input("生命周期(分钟)", min_value=-1, value=10, help="-1 代表永久")
            if st.form_submit_button("🚀 开启监测", use_container_width=True, type="primary"):
                with st.spinner("正在请求启动视频流..."):
                    payload = {"source": source, "lifetime_minutes": lifetime}
                    success, data, msg = api_request('POST', API_ENDPOINTS['STREAMS_START'], json=payload)
                if success and data:
                    st.toast(f"视频流任务已启动！ID: ...{data['stream_id'][-6:]}", icon="🚀")
                    st.session_state.viewing_stream_info = data
                    st.rerun()
                else:
                    st.error(f"启动失败: {msg}")

    # 显示当前正在观看的视频流
    if st.session_state.get("viewing_stream_info"):
        stream_info = st.session_state.viewing_stream_info
        st.subheader(f"正在播放: `{stream_info['source']}`")
        st.caption(f"Stream ID: `{stream_info['stream_id']}`")
        st.image(stream_info['feed_url'], caption=f"实时视频流 | 源: {stream_info['source']}")
    else:
        st.info("当前未选择任何视频流进行观看。请从下面的列表中选择一个，或启动一个新的监测任务。")

    st.divider()

    # 获取并显示所有活动的视频流列表
    st.subheader("所有活动中的监测任务")
    if st.button("刷新流列表"):
        st.cache_data.clear()  # 清除缓存以获取最新列表
        st.rerun()

    success, data, msg = api_request("GET", API_ENDPOINTS['STREAMS_LIST'])
    if not success:
        st.error(f"无法获取活动流列表: {msg}")
        return

    active_streams = data.get('streams', [])
    if not active_streams:
        st.info("目前没有正在运行的视频监测任务。")
    else:
        for stream in active_streams:
            stream_id = stream['stream_id']
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**来源:** `{stream['source']}`")
                    st.caption(
                        f"ID: `{stream_id}` | 启动于: {format_datetime_human(stream.get('started_at'))} | 将过期: {format_datetime_human(stream.get('expires_at'))}")
                with col2:
                    # 将按钮放在同一行
                    b_col1, b_col2 = st.columns(2)
                    if b_col1.button("👁️", key=f"view_{stream_id}", help="观看此流", use_container_width=True):
                        st.session_state.viewing_stream_info = stream
                        st.rerun()
                    if b_col2.button("⏹️", key=f"stop_{stream_id}", help="停止此流", type="primary",
                                     use_container_width=True):
                        with st.spinner(f"正在停止流 {stream['source']}..."):
                            endpoint = API_ENDPOINTS['STREAMS_STOP'].format(stream_id)
                            stop_success, _, stop_msg = api_request('POST', endpoint)
                        if stop_success:
                            st.toast(f"视频流 {stream['source']} 已停止。", icon="✅")
                            if st.session_state.viewing_stream_info and st.session_state.viewing_stream_info[
                                'stream_id'] == stream_id:
                                st.session_state.viewing_stream_info = None
                            st.rerun()
                        else:
                            st.error(f"停止失败: {stop_msg}")


# ==============================================================================
# 5. 主程序入口 (Main Application Entrypoint)
# ==============================================================================
def main():
    """主应用函数。"""
    initialize_session_state()
    render_sidebar()

    is_connected, _ = st.session_state.api_status
    # 仅在连接成功且数据尚未加载时自动加载
    if st.session_state.get("faces_data") is None and is_connected:
        refresh_all_data()

    pages = ["仪表盘", "人脸库管理", "实时监测"]
    st.session_state.active_page = st.radio(
        "主导航",
        options=pages,
        key="page_selector",
        label_visibility="collapsed",
        horizontal=True,
    )

    if st.session_state.active_page == "仪表盘":
        render_dashboard_page()
    elif st.session_state.active_page == "人脸库管理":
        render_management_page()
    elif st.session_state.active_page == "实时监测":
        render_monitoring_page()


if __name__ == "__main__":
    main()