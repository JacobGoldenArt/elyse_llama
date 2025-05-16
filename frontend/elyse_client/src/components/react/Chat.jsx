import { useChat } from '@ai-sdk/react'
import { useState } from 'react'
// import { Schema } from 'astro:schema' // Not directly used for API call structure here

// Backend /chat/stream expects a body like:
// {
//   "user_message": "string",
//   "session_id": "string", // Optional, defaults server-side if not sent
//   "settings": { // Optional, defaults server-side if not sent
//     "model": "string",
//     "temperature": number,
//     "max_tokens": number,
//     "top_p": number,
//     "frequency_penalty": number,
//     "presence_penalty": number
//   },
//   "initial_models_to_use": ["string"] // Optional, defaults server-side if not sent
// }


export default function () {
  const [modelsToUse, setModelsToUse] = useState(["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]) // Renamed for clarity
  const [modelSettings, setModelSettings] = useState({ // Renamed for clarity
    model: "gpt-4o-mini",
    temperature: 1,
    max_tokens: 1000,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0
  })
  const [sessionId, setSessionId] = useState("default_session"); // Manage session_id

  const { messages, input, handleInputChange, handleSubmit, data } = useChat({ // Added 'data' to access custom stream data
    api: 'http://localhost:8000/chat/stream',
    streamProtocol: 'data',
    // initialMessages: [], // You can load initial messages here if needed
  })

  // Custom submit handler to pass the correct body structure
  const customHandleSubmit = (e) => {
    e.preventDefault(); // Prevent default form submission
    handleSubmit(e, { // Pass event 'e' if your handleSubmit expects it
      body: {
        user_message: input, // The current text from the input field
        session_id: sessionId,
        settings: modelSettings,
        initial_models_to_use: modelsToUse,
      }
    });
  };
  
  // Example: Process custom data from the stream
  // console.log("Stream data:", data); // Check browser console for incoming custom data parts

  return (
    <form className="mt-12 flex w-full max-w-[300px] flex-col" onSubmit={customHandleSubmit}>
      {/* session_id is now managed in state, hidden input not strictly necessary for this setup */}
      {/* You could have an input to change session_id if needed:
      <label htmlFor="session_id_input">Session ID:</label>
      <input
        id="session_id_input"
        name="session_id"
        value={sessionId}
        onChange={(e) => setSessionId(e.target.value)}
        className="mt-1 rounded border px-2 py-1 outline-none focus:border-black"
      />
      */}

      <input
        id="userInput" // Changed id to avoid conflict if session_id input is added
        name="user_message_input" // name is for traditional forms, less critical here
        value={input} // Controlled input using `input` from useChat
        onChange={handleInputChange} // Use `handleInputChange` from useChat
        placeholder="What's your next question?"
        className="mt-3 rounded border px-2 py-1 outline-none focus:border-black"
      />
      
      <button className="mt-3 max-w-max rounded border px-3 py-1 outline-none hover:bg-black hover:text-white" type="submit">
        Ask &rarr;
      </button>
      <h3>Messages</h3>
      <div className="mt-3 border-t pt-3 min-h-[100px]">
        {messages.map((message, i) => (
          <div className="mt-3 border-l-2 border-gray-300 pl-3" key={message.id || i}> {/* Use message.id if available */}
            {message.content}
            {/* You'll need to handle rendering for custom data parts, like workflow steps or candidates, here */}
            {/* For example, checking message.experimental_streamData or a top-level `data` object from useChat */}
          </div>
        ))}
      </div>
    </form>
  )
}