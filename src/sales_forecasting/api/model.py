from pydantic import BaseModel, Field
from datetime import date
from typing import Optional


class RequestSchema(BaseModel):
    sales_date: date = Field(alias="Date")
    store: int = Field(alias="Store")
    sales: int = Field(alias="Sales")
    customers: Optional[int] = Field(alias="Customers")
    open_flag: int = Field(alias="Open")
    promo: int = Field(alias="Promo")
    day_of_week: int = Field(alias="DayOfWeek")
    state_holiday: Optional[str] = Field(alias="StateHoliday")
    school_holiday: Optional[int] = Field(alias="SchoolHoliday")
