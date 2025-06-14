from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from concurrent.futures import ProcessPoolExecutor
import uvicorn

# 假设你已有的函数定义都在这里，比如 is_equal 等
# 如果它们在其他模块中，请调整下面的导入路径
from math_utils import is_equal  # 请将 your_module 替换成实际模块名

app = FastAPI()

# 全局线程池执行器，用于 is_equal 内部的 run_in_executor
executor = ProcessPoolExecutor()

# 定义 POST 请求的请求体数据模型
class LaTeXInput(BaseModel):
    str1: str
    str2: str

@app.post("/is_equal")
async def check_latex_equality(input: LaTeXInput):
    try:
        # 调用你提供的异步函数 is_equal
        result = await is_equal(input.str1, input.str2, executor, math_mode="legacy")
        print(input.str1, "------", input.str2, "------", result)
        return {"equal": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=6284, reload=False)