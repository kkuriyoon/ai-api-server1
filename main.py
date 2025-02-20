from typing import Union
from fastapi import FastAPI
import pickle
import os

#model.py를 가져온다
from model import AndModel, XORModel, OrModel, NotModel


# API 서버를 생성
app = FastAPI()

and_model = AndModel()
or_model = OrModel()
not_model = NotModel()
xor_model = XORModel()


#endpoint 엔드포인트를 선언하며 GET으로 요청을 받고 경로는 /이다. 
@app.get("/")
def read_root():
    #딕려서니를 반환하면 JSON으로 직렬화된다.
    return {"Hello": "World"}


# AND 학습 & 예측
@app.post("/train/and")
def train_and():
    and_model.train()
    return {"result": "AND Model Trained"}

@app.get("/predict/and/{left}/{right}")
def predict_and(left: int, right: int):
    result = and_model.predict([left, right])
    return {"result": result}



# OR 학습 & 예측
@app.post("/train/or")
def train_or():
    or_model.train()
    return {"result": "OR Model Trained"}

@app.get("/predict/or/{left}/{right}")
def predict_or(left: int, right: int):
    result = or_model.predict([left, right])
    return {"result": result}



# NOT 학습 & 예측
@app.post("/train/not")
def train_not():
    not_model.train()
    return {"result": "NOT Model Trained"}

@app.get("/predict/not/{x}")
def predict_not(x: int):
    result = not_model.predict([x])
    return {"result": result}




# XOR 학습 & 예측 (PyTorch MLP)
@app.post("/train/xor")
def train_xor():
    xor_model.train()
    return {"result": "XOR Model Trained (PyTorch)"}

@app.get("/predict/xor/{left}/{right}")
def predict_xor(left: int, right: int):
    result = xor_model.predict([left, right])
    return {"result": result}



@app.get("/predict/{model_type}/{left}/{right}")
def predict(model_type: str, left: int, right: int):
    if model_type == "and":
        result = and_model.predict([left, right])
    elif model_type == "or":
        result = or_model.predict([left, right])
    elif model_type == "xor":
        result = xor_model.predict([left, right])
    elif model_type == "not":
        # NOT 연산은 입력이 1개면 충분하므로, left만 사용하고 right는 무시
        result = not_model.predict([left])
    else:
        return {"error": f"Unknown model type: {model_type}"}

    return {"model_type": model_type, "result": result}




'''
# 하나로 합친 엔드포인트
@app.get("/predict/{model_type}/{left}/{right}")
def predict(model_type: str, left: int, right: int):
    if model_type == "and":
        result = and_model.predict([left, right])
    elif model_type == "or":
        result = or_model.predict([left, right])
    elif model_type == "xor":
        result = xor_model.predict([left, right])
    else:
        return {"error": f"Unknown model type: {model_type}"}

    return {"model_type": model_type, "result": result}


# not을 위한 엔드포인트
@app.get("/predict/not/{x}")
def predict_not(x: int):
    result = not_model.predict([x])
    return {"model_type": "not", "result": result}

'''


'''
# 모델 저장: POST/save
@app.post("/save")
def save_model():
    #and_model을 피클로 저장
    with open("model.pkl", "wb") as f:
        pickle.dump(model,f)
    return {"message": "Model saved successfully"}



#모델 불러오기: POST/load
@app.post("/load")
def load_model():
    global and_model
    if not os.path.exists("and_model.pkl"):
        return {"error": "No pickle file found. Train and save first."}
    # 피클 파일을 열어서 and_model에 할당(역직렬화)
    with open("and_model.pkl", "rb") as f:
        and_model = pickle.load(f)
    return {"message": "Model loaded successfully."}

'''