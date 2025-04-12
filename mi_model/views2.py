from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .inference import predict_condition

@csrf_exempt
def classify_condition(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get("text", "")

            if not text:
                return JsonResponse({"error": "No text provided"}, status=400)

            prediction = predict_condition(text)
            return JsonResponse({"predicted_class": prediction})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
