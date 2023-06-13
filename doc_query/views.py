from django.shortcuts import render
import json
from rest_framework.viewsets import ViewSet
from rest_framework.views import APIView
from common import format_response,get_validation_error_message
from rest_framework.status import HTTP_400_BAD_REQUEST,HTTP_200_OK,HTTP_403_FORBIDDEN,HTTP_401_UNAUTHORIZED
# from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import AllowAny#,IsAuthenticated, IsAdminUser
from django.contrib.auth.models import User
from rest_framework import serializers


from django.http import JsonResponse,HttpResponse

from doc_query.aws_langchain.kendra_chat_open_ai import build_chain,run_chain


# chat_history = []
class StatusView(APIView):
    #APIViewExample
    #Example of a simple api which supports both get and post requests
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        data = {"status":"running"}
        return format_response(HTTP_200_OK,data=data)
        
    def post(self, request, format=None):
        # reads payload , modify and returns
        input_data = request.data
        input_data["foo"] = "bar"
        return format_response(HTTP_200_OK,data=input_data)

class SampleView(APIView):
    #Example post api with serializer validation
    class SampleInputSerializer(serializers.Serializer):
        # Define fields for input validation
        name = serializers.CharField(required=True)
        # fkey = serializers.PrimaryKeyRelatedField(required=False,queryset=ModelName.objects.all())
        amount = serializers.FloatField(required=False)
        # lat = serializers.DecimalField(required=True,max_digits=22,decimal_places=16)
        num_items = serializers.IntegerField(required=False,allow_null=True)
        is_new = serializers.BooleanField(required=True)
        
        class Meta:
            ref_name = "Sample View Input Serializer"

        def validate_type(self,val):
            #Custom Validations for the value in the key - type
            ALLOWED_VALUES = [1,2,3]
            # import constants from common.constants
            if val not in ALLOWED_VALUES:
                raise serializers.ValidationError({"field":"type",
                                                    "error":f"type must be in one of {ALLOWED_VALUES}"
                                                    })
            return val

        def validate(self,attrs):
            #do some more validations
            amount = attrs.get("amount")
            name = attrs["name"]
            # if not something:
            #     raise serializers.ValidationError()
            return attrs

    serializer_class = SampleInputSerializer
    permission_classes = (AllowAny,)
    def post(self,request):
        # user = request.user
        serializer = self.SampleInputSerializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
        except serializers.ValidationError as e:
            return format_response(HTTP_400_BAD_REQUEST,message=get_validation_error_message(e.detail),errors=e.detail)
        data = serializer.validated_data
        #Do Something
        return format_response(HTTP_200_OK,data=data,message="Done")



class chatbot(APIView):
    def __init__(self):
        self.qa = build_chain()
        

    def post(self, request):
        if request.method == "POST":
            
            data = json.loads(request.body)
            
            query = data.get("query")
            
            result = run_chain(self.qa, query)
            
            # response_data = {'ans':result['answer']}
       
            # chat_history.append((query,result['answer']))
            # if len(chat_history) >=3:
            #     chat_history[-3:-1]
            # print(chat_history)
            # result = result.split('```json\n')[1].split('\n```')[0]
            res = result['answer'].split('```json\n')[1].split('\n```')[0]
            return HttpResponse(res)
