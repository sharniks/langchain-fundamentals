from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):

    name: str = 'nikhil'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5,
                        description='cgpa of student')


new_student = {'age': '32', 'email': 'abc@gmail.com'}

student = Student(**new_student)

print(student)

student_dict = dict(student)

student_json = student.model_dump_json()