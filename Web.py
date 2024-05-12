from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from typing import Annotated
from .models import MoscowModel, PeterModel


app = FastAPI()
templates = Jinja2Templates(directory="RealEstateCostCalculator/templates")
moscow_prediction_model = MoscowModel()
peter_prediction_model = PeterModel()


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse(
        request=request, name="main.html", context={"id": id}
    )


@app.post("/result")
async def result(request: Request, city: Annotated[str, Form()], floors_count: Annotated[int, Form()],
                 street: Annotated[str, Form()], house: Annotated[str, Form()], district: Annotated[str, Form()],
                 floor: Annotated[int, Form()], year_of_construction: Annotated[int, Form()],
                 rooms: Annotated[int, Form()], total_square: Annotated[float, Form()],
                 kitchen_square: Annotated[float, Form()] = -1, living_square: Annotated[float, Form()] = -1):
    city = city.lower()
    street = street.lower()
    house = house.lower()
    district = district.lower()
    avg_living_square, avg_kitchen_square = living_square, kitchen_square
    if living_square == -1:
        avg_living_square = 0.75 * total_square
    if kitchen_square == -1:
        avg_kitchen_square = 0.2 * total_square

    if city == "москва":
        prediction = moscow_prediction_model.calculate(
            floors_count, floor, street, house, district, year_of_construction,
            avg_living_square, avg_kitchen_square, total_square, rooms
        )
    elif city == "санкт-петербург":
        prediction = peter_prediction_model.calculate(
            floors_count, floor, street, house, district, year_of_construction,
            avg_living_square, avg_kitchen_square, total_square, rooms
        ) * 1.5
    prediction1 = round(prediction / 10**6, 1)
    prediction_range1 = round(prediction * 0.93 / 10 ** 6, 1)
    prediction_range2 = round(prediction * 1.07 / 10 ** 6, 1)
    print(prediction, prediction_range1, prediction_range2)  # instead of logger
    return templates.TemplateResponse(
        request=request, name="result.html",
        context={"city": city, "floors_count": floors_count, "floor": floor, "street": street, "house": house,
                 "district": district, "year_of_construction": year_of_construction, "rooms": rooms,
                 "total_square": total_square, "kitchen_square": kitchen_square, "living_square": living_square,
                 "prediction": prediction1, "prediction_range1": prediction_range1,
                 "prediction_range2": prediction_range2
                 }
    )


@app.get("/api_guide")
async def api_guide(request: Request):
    return templates.TemplateResponse(request=request, name="api_guide.html")


@app.get("/v1")
async def v1(request: Request, city: str, street: str, house: str, district: str, floor: int, rooms: int,
             total_square: float, floors_count: int = -1, kitchen_square: float = -1, living_square: float = -1,
             year_of_construction: int = -1):
    city = city.lower()
    street = street.lower()
    house = house.lower()
    district = district.lower()
    if city != "москва" and city != "санкт-петербург":
        return {"result": -1, "error": 400, "reason": "Unsupported city"}
    if floors_count <= 0 and floors_count != -1:
        return {"result": -1, "error": 400, "reason": f"Invalid value of floors_count: expected >= 1, got {floors_count}"}
    if kitchen_square <= 0 and kitchen_square != -1:
        return {"result": -1, "error": 400, "reason": f"Invalid value of kitchen_square: expected > 0, got {kitchen_square}"}
    if living_square <= 0 and living_square != -1:
        return {"result": -1, "error": 400, "reason": f"Invalid value of living_square: expected > 0, got {living_square}"}
    if rooms <= 0:
        return {"result": -1, "error": 400, "reason": f"Invalid value of rooms: expected >= 1, got {rooms}"}
    if total_square <= 0:
        return {"result": -1, "error": 400, "reason": f"Invalid value of total_square: expected > 0, got {total_square}"}
    if year_of_construction < 1700 and year_of_construction != -1:
        return {"result": -1, "error": 400, "reason": f"Invalid value of year_of_construction: expected >= 1700, got {year_of_construction}"}

    # fill optional parameters with average values if they are not in request
    if living_square == -1:
        living_square = 0.75 * total_square
    if kitchen_square == -1:
        kitchen_square = 0.2 * total_square
    if floors_count == -1:
        floors_count = 10
    if year_of_construction == -1:
        year_of_construction = 2000
    if city == "москва":
        prediction = moscow_prediction_model.calculate(
            floors_count, floor, street, house, district, year_of_construction, living_square, kitchen_square,
            total_square, rooms
        )
    elif city == "санкт-петербург":
        prediction = peter_prediction_model.calculate(
            floors_count, floor, street, house, district, year_of_construction, living_square, kitchen_square,
            total_square, rooms
        ) * 1.5
    return {"result": prediction}


@app.exception_handler(404)
async def custom_404_handler(request, __):
    return templates.TemplateResponse(request=request, name="error_page.html", context={"error": "Страница не найдена"})


@app.exception_handler(500)
async def custom_500_handler(request, __):
    return templates.TemplateResponse(request=request, name="error_page.html", context={"error": "Ошибка в работе сервиса"})
