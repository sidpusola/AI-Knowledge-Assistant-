import { useState } from 'react'
import "./App.css"
function App(){
  const[file,setFile]=useState(null);
  const[question,setQuestion]=useState("");
  const[answer,setAnswer]=useState("");
  const[uploadMessage,setUploadMessage]=useState("");
  const[loading, setLoading]=useState(false);
  const[uploading,setUploading]=useState(false);

  const uploadFile=async()=>{           //communicating with backend takes time
    if(!file){
      setUploadMessage("Please select a file.");
      return;
    }
    setUploading(true);

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
    }finally{
      setUploading(false);
    }
  };
  
  const askQuestion=async()=>{
    if(!question.trim()){
      return;
    }
    setLoading(true)

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
      setQuestion("");
    
    }catch(error){
      setAnswer("Could not connect to backend")
    }finally{
      setLoading(false)
    }
  };

  return (
    <div className='app'>
      <h1>AI Knowledge Assistant</h1>

      <section className='card'> 
        <h2>Upload Documents</h2>

        <input
         type="file"
         onChange={(event)=>setFile(event.target.files[0])}
        />

        <button onClick={uploadFile} disabled={uploading}>
          {uploading ? "Uplaoding...":"Upload"}
        </button>

        <p className='status'>{uploadMessage}</p>
      </section>

      <hr />

      <section className='card'>
        <h2>Ask a Question</h2>

        <input
         type="text"
         value={question}
         placeholder='Ask something from your document' 
         onChange={(event)=>setQuestion(event.target.value)}
         onKeyDown={(event)=>{
          if(event.key=="Enter"){
            askQuestion();
          }
         }
        }
        />

        <button onClick={askQuestion} disabled={loading}>
          {loading? "Generating...":"Ask"}
          
        </button>

        <div className='answer-box'>
        <h3>Answer</h3>
        {loading ? <p>Generating answer...</p> : <p>{answer}</p>}
        </div>
      </section>
    </div>
  );
}

export default App;