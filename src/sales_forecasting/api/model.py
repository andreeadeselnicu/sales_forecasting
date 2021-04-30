from pydantic import BaseModel, Field
from datetime import date


class RequestBody(BaseModel):
    sale_date: date = Field(alias="Date")
    store: int = Field(alias="Store")
    sales: int = Field(alias="Sales")
    customers: int = Field(alias="Customers")
    open_flag: int = Field(alias="Open")
    promo: int = Field(alias="Promo")
    day_of_week: int = Field(alias="DayOfWeek")
    state_holiday: int = Field(alias="StateHoliday")
    school_holiday: int = Field(alias="SchoolHoliday")

    


    