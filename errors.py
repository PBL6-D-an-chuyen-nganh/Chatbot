class ChatbotError(Exception):
    def __init__(self, message: str, error_type: str = "ChatbotError"):
        self.message = message
        self.error_type = error_type
        super().__init__(message)

class ResourceNotReadyError(ChatbotError):
    def __init__(self, message: str = "Resource not ready"):
        super().__init__("Hệ thống tra cứu bệnh hiện chưa sẵn sàng. Vui lòng thử lại sau", error_type="not_found")

class ImageProcessingError(ChatbotError):
    def __init__(self, reason: str):
        super().__init__(
            f"Không thể xử lý ảnh: {reason}",
            "validation"
        )

class AppointmentError(ChatbotError):
    def __init__(self, reason: str):
        super().__init__(
            f"Lỗi đặt lịch: {reason}",
            "server"
        )
