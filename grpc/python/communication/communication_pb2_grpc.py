# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import communication_pb2 as communication__pb2


class GRPCServerStub(object):
    """***************************************************************************
    * Service definition:
    * - notation:
    *   rpc <service-name> (<client-request-ms>) returns (<server-response-msg>)
    *
    * - mode: 
    *   - unary-unary             -> void, void
    *   - client-streaming        -> stream, void
    *   - server-streaming        -> void, stream
    *   - bidirectional streaming -> stream, stream
    ***************************************************************************
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Test = channel.unary_unary(
                '/communication.GRPCServer/Test',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseEmpty.FromString,
                )
        self.Setup = channel.unary_unary(
                '/communication.GRPCServer/Setup',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseCylinderPos.FromString,
                )
        self.Execute = channel.unary_unary(
                '/communication.GRPCServer/Execute',
                request_serializer=communication__pb2.RequestDoAction.SerializeToString,
                response_deserializer=communication__pb2.ResponseDistance.FromString,
                )
        self.DistanceRobotCylinder = channel.unary_unary(
                '/communication.GRPCServer/DistanceRobotCylinder',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseDistance.FromString,
                )
        self.RobotReset = channel.unary_unary(
                '/communication.GRPCServer/RobotReset',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseEmpty.FromString,
                )
        self.GetRobotJointStates = channel.unary_unary(
                '/communication.GRPCServer/GetRobotJointStates',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseRobotJointStates.FromString,
                )
        self.GetRobotPosition = channel.unary_unary(
                '/communication.GRPCServer/GetRobotPosition',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseRobotPosition.FromString,
                )
        self.RobotAct = channel.unary_unary(
                '/communication.GRPCServer/RobotAct',
                request_serializer=communication__pb2.RequestDoAction.SerializeToString,
                response_deserializer=communication__pb2.ResponseEmpty.FromString,
                )
        self.RobotIsStable = channel.unary_unary(
                '/communication.GRPCServer/RobotIsStable',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseBinaryStatus.FromString,
                )
        self.RobotCheckCollision = channel.unary_unary(
                '/communication.GRPCServer/RobotCheckCollision',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseBinaryStatus.FromString,
                )
        self.CylinderReset = channel.unary_unary(
                '/communication.GRPCServer/CylinderReset',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseEmpty.FromString,
                )
        self.CylinderRandomReset = channel.unary_unary(
                '/communication.GRPCServer/CylinderRandomReset',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseEmpty.FromString,
                )
        self.CylinderGetPosition = channel.unary_unary(
                '/communication.GRPCServer/CylinderGetPosition',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseCylinderPos.FromString,
                )
        self.CylinderIsStable = channel.unary_unary(
                '/communication.GRPCServer/CylinderIsStable',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseBinaryStatus.FromString,
                )
        self.CylinderIsOnGround = channel.unary_unary(
                '/communication.GRPCServer/CylinderIsOnGround',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseBinaryStatus.FromString,
                )
        self.CameraGetImage = channel.unary_unary(
                '/communication.GRPCServer/CameraGetImage',
                request_serializer=communication__pb2.RequestEmpty.SerializeToString,
                response_deserializer=communication__pb2.ResponseCameraImage.FromString,
                )


class GRPCServerServicer(object):
    """***************************************************************************
    * Service definition:
    * - notation:
    *   rpc <service-name> (<client-request-ms>) returns (<server-response-msg>)
    *
    * - mode: 
    *   - unary-unary             -> void, void
    *   - client-streaming        -> stream, void
    *   - server-streaming        -> void, stream
    *   - bidirectional streaming -> stream, stream
    ***************************************************************************
    """

    def Test(self, request, context):
        """experiment
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Setup(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Execute(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DistanceRobotCylinder(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RobotReset(self, request, context):
        """robot
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRobotJointStates(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRobotPosition(self, request, context):
        """rpc GetRobotCurrentFrame (RequestImage) returns stream ()
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RobotAct(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RobotIsStable(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RobotCheckCollision(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CylinderReset(self, request, context):
        """cylinder
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CylinderRandomReset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CylinderGetPosition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CylinderIsStable(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CylinderIsOnGround(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CameraGetImage(self, request, context):
        """camera
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GRPCServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Test': grpc.unary_unary_rpc_method_handler(
                    servicer.Test,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseEmpty.SerializeToString,
            ),
            'Setup': grpc.unary_unary_rpc_method_handler(
                    servicer.Setup,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseCylinderPos.SerializeToString,
            ),
            'Execute': grpc.unary_unary_rpc_method_handler(
                    servicer.Execute,
                    request_deserializer=communication__pb2.RequestDoAction.FromString,
                    response_serializer=communication__pb2.ResponseDistance.SerializeToString,
            ),
            'DistanceRobotCylinder': grpc.unary_unary_rpc_method_handler(
                    servicer.DistanceRobotCylinder,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseDistance.SerializeToString,
            ),
            'RobotReset': grpc.unary_unary_rpc_method_handler(
                    servicer.RobotReset,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseEmpty.SerializeToString,
            ),
            'GetRobotJointStates': grpc.unary_unary_rpc_method_handler(
                    servicer.GetRobotJointStates,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseRobotJointStates.SerializeToString,
            ),
            'GetRobotPosition': grpc.unary_unary_rpc_method_handler(
                    servicer.GetRobotPosition,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseRobotPosition.SerializeToString,
            ),
            'RobotAct': grpc.unary_unary_rpc_method_handler(
                    servicer.RobotAct,
                    request_deserializer=communication__pb2.RequestDoAction.FromString,
                    response_serializer=communication__pb2.ResponseEmpty.SerializeToString,
            ),
            'RobotIsStable': grpc.unary_unary_rpc_method_handler(
                    servicer.RobotIsStable,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseBinaryStatus.SerializeToString,
            ),
            'RobotCheckCollision': grpc.unary_unary_rpc_method_handler(
                    servicer.RobotCheckCollision,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseBinaryStatus.SerializeToString,
            ),
            'CylinderReset': grpc.unary_unary_rpc_method_handler(
                    servicer.CylinderReset,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseEmpty.SerializeToString,
            ),
            'CylinderRandomReset': grpc.unary_unary_rpc_method_handler(
                    servicer.CylinderRandomReset,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseEmpty.SerializeToString,
            ),
            'CylinderGetPosition': grpc.unary_unary_rpc_method_handler(
                    servicer.CylinderGetPosition,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseCylinderPos.SerializeToString,
            ),
            'CylinderIsStable': grpc.unary_unary_rpc_method_handler(
                    servicer.CylinderIsStable,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseBinaryStatus.SerializeToString,
            ),
            'CylinderIsOnGround': grpc.unary_unary_rpc_method_handler(
                    servicer.CylinderIsOnGround,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseBinaryStatus.SerializeToString,
            ),
            'CameraGetImage': grpc.unary_unary_rpc_method_handler(
                    servicer.CameraGetImage,
                    request_deserializer=communication__pb2.RequestEmpty.FromString,
                    response_serializer=communication__pb2.ResponseCameraImage.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'communication.GRPCServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GRPCServer(object):
    """***************************************************************************
    * Service definition:
    * - notation:
    *   rpc <service-name> (<client-request-ms>) returns (<server-response-msg>)
    *
    * - mode: 
    *   - unary-unary             -> void, void
    *   - client-streaming        -> stream, void
    *   - server-streaming        -> void, stream
    *   - bidirectional streaming -> stream, stream
    ***************************************************************************
    """

    @staticmethod
    def Test(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/Test',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseEmpty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Setup(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/Setup',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseCylinderPos.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Execute(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/Execute',
            communication__pb2.RequestDoAction.SerializeToString,
            communication__pb2.ResponseDistance.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DistanceRobotCylinder(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/DistanceRobotCylinder',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseDistance.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RobotReset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/RobotReset',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseEmpty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetRobotJointStates(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/GetRobotJointStates',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseRobotJointStates.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetRobotPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/GetRobotPosition',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseRobotPosition.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RobotAct(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/RobotAct',
            communication__pb2.RequestDoAction.SerializeToString,
            communication__pb2.ResponseEmpty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RobotIsStable(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/RobotIsStable',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseBinaryStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RobotCheckCollision(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/RobotCheckCollision',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseBinaryStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CylinderReset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/CylinderReset',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseEmpty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CylinderRandomReset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/CylinderRandomReset',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseEmpty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CylinderGetPosition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/CylinderGetPosition',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseCylinderPos.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CylinderIsStable(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/CylinderIsStable',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseBinaryStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CylinderIsOnGround(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/CylinderIsOnGround',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseBinaryStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CameraGetImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.GRPCServer/CameraGetImage',
            communication__pb2.RequestEmpty.SerializeToString,
            communication__pb2.ResponseCameraImage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
