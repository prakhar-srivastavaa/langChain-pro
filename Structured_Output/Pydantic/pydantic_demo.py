from pydantic import BaseModel,EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    email: Optional[EmailStr]= None
    CGPA: float=Field(gt=0,lt=10, default=5, description="A decimal value of total scored marks as CGPA out of 10")

new_student={'name':'Prakhar','email':'abc@gmail.com', 'CGPA':8.07}
new_number={'name':'Srivastava'}
student= Student(**new_student)
number= Student(**new_number)

print(number)
print(student)