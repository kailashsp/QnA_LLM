from django.urls import path,include
from doc_query import views
urlpatterns = [
    path("status/", views.StatusView.as_view(), name="api_view"),
    path("sample/", views.SampleView.as_view(), name="sample_view"),
    path("chatbot/", views.chatbot.as_view(), name="chatbot"),
    path("poechat/",views.poechat.as_view(),name="claude")
]