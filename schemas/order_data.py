from pydantic import BaseModel

class OrderData(BaseModel):
    days_since_last_order:int
    shipping_duration_days:int
    used_coupon : int
    order_amount : float