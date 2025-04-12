# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from .inference import predict_condition

# @csrf_exempt
# def classify_condition(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             text = data.get("text", "")

#             if not text:
#                 return JsonResponse({"error": "No text provided"}, status=400)

#             prediction = predict_condition(text)
#             return JsonResponse({"predicted_class": prediction})

#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)

#     return JsonResponse({"error": "Invalid request method"}, status=405)

# from django.shortcuts import render
# from .model_loader import predict_condition  # adjust import as needed




# def predict_disease_view(request):
#     prediction = None
#     if request.method == 'POST':
#         symptoms = request.POST.get('symptoms', '')
#         prediction = predict_condition(symptoms)
    
#     return render(request, 'predict.html', {'prediction': prediction})





from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .model_loader import predict_condition  # Make sure this path is correct


# 1️⃣ - This handles the form in services.html
# def classify_condition(request):
#     prediction = None
#     if request.method == 'POST':
#         user_input = request.POST.get('user_input', '')
#         if user_input:
#             prediction = predict_condition(user_input)

#     # Renders the same services.html with prediction result
#     return render(request, 'services.html', {'prediction': prediction})
def classify_condition(request):
    prediction = None
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        print("📝 User input:", user_input)  # Debug log
        if user_input:
            prediction = predict_condition(user_input)
            print("🔍 Prediction:", prediction)  # Debug log

    return render(request, 'services.html', {'prediction': prediction})



# 2️⃣ - Optional: Keep this if you're using an API endpoint via JSON
@csrf_exempt
def classify_condition_api(request):
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

