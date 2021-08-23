from fastapi import HTTPException


class HTTPFileNotFoundException(HTTPException):
    def __init__(self, filename):
        super(HTTPFileNotFoundException, self).__init__(
            status_code=404, detail=f"File not found: {filename}."
        )
