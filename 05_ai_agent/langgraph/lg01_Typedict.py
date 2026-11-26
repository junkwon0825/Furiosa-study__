from typing import TypedDict, Dict

sample_dict: Dict[str, str] = {
    'name' : '김말똥',
    'age' : 20,
    'job' : '회사원'
}

# print(sample_dict)  # Output: {'name': '김말똥', 'age': 20, 'job': '회사원'}
# print(sameple_dict['name'])  # Output: 김말똥

###################################################
class Person(TypedDict):
    name: str
    age: int
    job: str

person = Person(name='김말똥', age=20, job='회사원')
print(person)  # Output: {'name': '김말똥', 'age': 20, 'job': '회사원'} 
#######################################################

person2: Person = { 'name': '김말똥', 'age': 20, 'job': '회사원' }
print(person2)  # Output: {'name': '김말똥', 'age': 20, 'job': '회사원'} 
