
import os,sys,io,traceback
from typing import TypedDict,List,Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI

app=FastAPI(title="AI Developer & Tester")

key=os.getenv("GEMINI_API_KEY")
if not key: raise ValueError("GEMINI_API_KEY not set")

llm=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=key
)

class CrewState(TypedDict):
    messages:List[BaseMessage]
    next_step:Optional[str]
    code:Optional[str]
    report:Optional[str]

class TaskRequest(BaseModel):
    task:str

def text(x):
    if isinstance(x,str): return x
    if isinstance(x,list):
        return "\n".join(
            str(i.get("text","")) if isinstance(i,dict) else str(i)
            for i in x
        )
    return str(x)

@tool
def run_python_code(code:str)->str:
    """Execute Python code and return output or error."""
    old=sys.stdout
    sys.stdout=io.StringIO()
    try:
        exec(
            code.replace("```python","").replace("```","").strip(),
            {},{}
        )
        out=sys.stdout.getvalue()
    except Exception:
        out="Execution Error:\n"+traceback.format_exc()
    finally:
        sys.stdout=old
    return out.strip() or "Success (no terminal output)"

@tool
def generate_test_cases(task:str)->str:
    """Generate test scenarios for the coding task."""
    p=f"""Generate 3 to 5 specific test scenarios for this coding task:
'{task}'. Include standard and edge cases. Return numbered list."""
    return text(llm.invoke(p).content)

def task_input(s):
    return {
        "messages":[
            HumanMessage(content=s["messages"][-1].content)
        ],
        "next_step":"developer"
    }

def developer(s):
    task=s["messages"][-1].content
    p=f"Write a clean Python script to solve this: {task}. Only return code."
    code=text(llm.invoke(p).content)
    code=code.replace("```python","").replace("```","").strip()
    return {"code":code,"next_step":"tester"}

def tester(s):
    tests=generate_test_cases.invoke(s["messages"][-1].content)
    output=run_python_code.invoke({"code":s["code"]})
    return {
        "report":
        f"EXECUTION OUTPUT:\n{output}\n\nTEST SCENARIOS:\n{tests}",
        "next_step":"manager"
    }
def manager(s):
    print("\n[MANAGER]\n",s.get("report",""))
    return {"next_step":"archiver"}

def archiver(s):
    return {"next_step":"exit"}

g=StateGraph(CrewState)

for name,fn in [
    ("task_input",task_input),
    ("developer",developer),
    ("tester",tester),
    ("manager",manager),
    ("archiver",archiver)
]:
    g.add_node(name,fn)

g.add_edge(START,"task_input")
g.add_edge("task_input","developer")
g.add_edge("developer","tester")
g.add_edge("tester","manager")
g.add_edge("manager","archiver")
g.add_edge("archiver",END)

rt_app=g.compile()

@app.get("/",response_class=HTMLResponse)
def home():
    return """
<html>
<head>
<title>AI Developer & Tester</title>
<style>
body{
font-family:Arial;
background:#0f172a;
color:white;
max-width:900px;
margin:auto;
padding:40px
}
.card{
background:#1e293b;
padding:20px;
margin:20px 0;
border-radius:10px
}
textarea{
width:100%;
height:120px;
padding:10px;
box-sizing:border-box
}
button{
width:100%;
padding:12px;
margin-top:10px;
background:#2563eb;
color:white;
border:0;
border-radius:6px
}
pre{
background:#020617;
padding:15px;
white-space:pre-wrap
}
</style>
</head>

<body>

<h1>🤖 AI Developer & Tester</h1>

<div class="card">
<h2>Enter Coding Task</h2>

<textarea id="task"
placeholder="Example: Write a Python program to check whether a number is prime.">
</textarea>

<button onclick="run()">Run Workflow</button>
</div>

<div id="result"></div>

<script>
async function run(){

let task=document.getElementById("task").value.trim();

if(!task)
return alert("Please enter a coding task.");

document.getElementById("result").innerHTML=
"<p>⏳ AI is working...</p>";

try{

let r=await fetch("/run",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({task})
});

let d=await r.json();
if(d.status=="success"){
document.getElementById("result").innerHTML=
'<div class="card"><h2>Generated Code</h2><pre>'+
d.generated_code.replace(/</g,"&lt;")+
'</pre></div>'+
'<div class="card"><h2>Test Report</h2><pre>'+
d.report.replace(/</g,"&lt;")+
'</pre></div>';
}else{
document.getElementById("result").innerHTML=
"<p>"+d.message+"</p>";

}
}catch(e){
document.getElementById("result").innerHTML=
"<p>Error: "+e.message+"</p>";
}
}
</script>
</body>
</html>
"""
@app.post("/run")
def run(request:TaskRequest):

    task=request.task.strip()

    if not task:
        return {
            "status":"error",
            "message":"Task cannot be empty."
        }
    try:
        state=rt_app.invoke({
            "messages":[HumanMessage(content=task)],
            "next_step":"task_input",
            "code":None,
            "report":None
        })
        return {
            "status":"success",
            "generated_code":state["code"],
            "report":state["report"]
        }
    except Exception as e:

        print(traceback.format_exc())
        return {
            "status":"error",
            "message":str(e)
        }
if __name__=="__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT",8000))
    )
