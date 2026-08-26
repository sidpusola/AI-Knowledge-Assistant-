import { useState } from 'react'
function app(){
  const[file,setFile]=useState(null);
  const[question,setQuestion]=useState("");
  const[answer,setAnswer]=useState("");
  const[uploadMessage,setUploadMessage]=useState("");

  const uploadFile=async()=>{           //communicating with backend takes time
    if(!file){
      setUploadMessage("Please select a file.");
      return;
    }

    const formData=new FormData();   //used to send files through HTTP to backend
    formData.append("file",file);
    

    try{
      const response=await fetch("http://127.0.0.1:8000/upload",{         //sending file with fetch
        method:"POST",
        body:formData,
      });
      
      const data= await response.json();  

      if(!response.ok){
        setUploadMessage(data.detail || "upload failed");
        return;
      }

      setUploadMessage(data.message);
    }catch(error){
      setUploadMessage("Could not connect to backend");
    }
  };
  
  const askQuestion=async()=>{
    if(!question.trim()){
      return;
    }

    try{
      const response= await fetch("http://127.0.0.1:8000/ask",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({question:question})
      });

      const data=await response.json();

      if(!response.ok){
        setAnswer(data.detail || "Something went wrong");
        return;
      }

      setAnswer(data.answer);
    
    }catch(error){
      setAnswer("Could not connect to backend")
    }
  };

  return (
    <div>
      <h1>AI Knowledge Assistant</h1>

      <section> 
        <h2>Upload Documents</h2>

        <input
         type="file"
         onChange={(event)=>setFile(event.target.files[0])}
        />

        <button onClick={uploadFile}>
          Upload 
        </button>
        <p>{uploadMessage}</p>
      </section>

      <hr />

      <section>
        <h2>Ask a Question</h2>

        <input
         type="text"
         value={question}
         placeholder='Ask something from your document' 
         onChange={(event)=>setQuestion(event.target.value)}
        />

        <button onClick={askQuestion}>
          Ask
        </button>

        <h3>Answer</h3>

        <p>{answer}</p>
      </section>
    </div>
  );
}

export default app;