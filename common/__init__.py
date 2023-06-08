from common.constants import REQUEST_FAILED, REQUEST_STATUSES, REQUEST_SUCCEEDED
from rest_framework.response import Response
from rest_framework import status

def format_response(resp_status,req_status=None,data=None,message="",warnings="",display_message="",errors={}):
    if not req_status:
        if resp_status == status.HTTP_200_OK:
            req_status = REQUEST_SUCCEEDED
        if resp_status == status.HTTP_400_BAD_REQUEST:
            req_status = REQUEST_FAILED
    assert(req_status in REQUEST_STATUSES)
    display_message = message if message else ""
    resp =  {
        "status" : req_status
    }
    if data!=None:
        resp["data"] = data
    if message:
        resp["message"] = message
    if errors:
        resp["errors"] = errors
    if display_message:
        resp["display_message"] = display_message
    if warnings:
        resp["warnings"] = warnings
    return Response(resp,status=resp_status)

def get_validation_error_message(d):
    return "; ".join([k+" - "+"".join(v) for k,v in d.items()])