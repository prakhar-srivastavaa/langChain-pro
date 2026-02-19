from pydantic import BaseModel,EmailStr
from typing import Optional

class Student(BaseModel):
    name: str
    email: Optional[EmailStr]= None


new_student={'name':'Prakhar','email':'abc@gmail.com'}
new_number={'name':'Srivastava'}
student= Student(**new_student)
number= Student(**new_number)

print(number)
print(student)