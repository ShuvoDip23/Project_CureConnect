# from django.urls import path
# from .views import classify_condition

# urlpatterns = [
#     path("classify/", classify_condition, name="classify_condition"),
# ]
 # mlmodel/urls.py
from django.urls import path
from .views import classify_condition

urlpatterns = [
    path('predict/', classify_condition, name='predict_condition'),
]
