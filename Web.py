from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from typing import Annotated
from .models import MoscowModel


app = FastAPI()
templates = Jinja2Templates(directory="RealEstateCostCalculator/templates")
# moscow_prediction_model = MoscowModel()


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse(
        request=request, name="main.html", context={"id": id}
    )


@app.post("/result")  # TODO среднее значение необязательных параметров чтобы поставить их в default
async def result(request: Request, city: Annotated[str, Form()], floors_count: Annotated[int, Form()],
                 street: Annotated[str, Form()], house: Annotated[str, Form()], district: Annotated[str, Form()],
                 floor: Annotated[int, Form()], year_of_construction: Annotated[int, Form()],
                 rooms: Annotated[int, Form()], total_meters: Annotated[float, Form()],
                 kitchen_metr: Annotated[float, Form()] = -1, living_metr: Annotated[float, Form()] = -1):
    avg_living_metr, avg_kitchen_metr = living_metr, kitchen_metr
    if living_metr == -1:
        avg_living_metr = 0.75 * total_meters
    if kitchen_metr == -1:
        avg_kitchen_metr = 0.2 * total_meters
    prediction = 1
    # prediction = moscow_prediction_model.calculate(floors_count, floor, street, house, district, year_of_construction,
    #                                                avg_living_metr, avg_kitchen_metr, total_meters, rooms)  # todo проверить что точно правильный порядок передачи параметров
    prediction1 = round(prediction / 10**6, 1)
    prediction_range1 = round(prediction * 0.93 / 10 ** 6, 1)
    prediction_range2 = round(prediction * 1.07 / 10 ** 6, 1)
    print(prediction, prediction_range1, prediction_range2)
    return templates.TemplateResponse(
        request=request, name="result.html",
        context={"city": city, "floors_count": floors_count, "floor": floor, "street": street, "house": house,
                 "district": district, "year_of_construction": year_of_construction, "rooms": rooms,
                 "total_meters": total_meters, "kitchen_metr": kitchen_metr, "living_metr": living_metr,
                 "prediction": prediction1, "prediction_range1": prediction_range1,
                 "prediction_range2": prediction_range2
                 }
    )


# @app.get("/api_guide")
# async def api_guide():
#     return
