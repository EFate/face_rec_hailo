import msgpack_numpy
from socketio import packet


class MsgPackNumpyPacket(packet.Packet):
    uses_binary_events = False

    def encode(self):
        """Encode the packet for transmission."""
        return msgpack_numpy.dumps(self._to_dict())

    def decode(self, encoded_packet):
        """Decode a transmitted package."""
        decoded = msgpack_numpy.loads(encoded_packet)
        self.packet_type = decoded["type"]
        self.data = decoded.get("data")
        self.id = decoded.get("id")
        self.namespace = decoded["nsp"]
