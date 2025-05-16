import axios from 'axios'
import { useState } from 'react'
 
export default function () {
  const [messages, setMessages] = useState('')
  return (
    <form
      className="mt-12 flex w-full max-w-[300px] flex-col"
      onSubmit={(e) => {
        e.preventDefault()
        axios.post('http://localhost:8000/update', {
          messages,
        })
      }}
    >
      <textarea
        value={messages}
        id="learn_messages"
        name="learn_messages"
        onChange={(e) => setMessages(e.target.value)}
        placeholder="Things to learn [seperated by comma (,)]"
        className="mt-3 rounded border px-2 py-1 outline-none focus:border-black"
      />
      <button className="mt-3 max-w-max rounded border px-3 py-1 outline-none hover:bg-black hover:text-white" type="submit">
        Learn &rarr;
      </button>
    </form>
  )
}