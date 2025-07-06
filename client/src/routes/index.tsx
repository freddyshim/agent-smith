import { createFileRoute } from '@tanstack/react-router'
import { useMutation } from '@tanstack/react-query'
import { useState, useRef, useEffect } from 'react'

const API_URL = 'http://localhost:8000'

export const Route = createFileRoute('/')({
  component: App,
})

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState('')
  const [chatInput, setChatInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendChatMessage = async (prompt: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: prompt,
    }

    const assistantMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
    }

    setMessages((prev) => [...prev, userMessage, assistantMessage])
    setIsStreaming(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
        }),
      })

      if (res.ok && res.body) {
        const reader = res.body.getReader()
        const decoder = new TextDecoder()

        while (true) {
          const { value, done } = await reader.read()
          if (done) {
            break
          }
          const decodedChunk = decoder.decode(value, { stream: true })

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessage.id
                ? { ...msg, content: msg.content + decodedChunk }
                : msg,
            ),
          )
        }
      } else {
        throw new Error('failed to fetch data')
      }
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? { ...msg, content: 'Error: Failed to get response' }
            : msg,
        ),
      )
    } finally {
      setIsStreaming(false)
    }
  }

  const { mutateAsync: uploadDocument } = useMutation({
    mutationKey: ['document'],
    mutationFn: async (file: File) => {
      const formData = new FormData()
      formData.append('file', file)
      
      const res = await fetch(`${API_URL}/document`, {
        method: 'POST',
        body: formData,
      })
      
      if (res.ok) {
        return res.json()
      } else {
        const errorData = await res.json()
        throw new Error(errorData.detail || 'failed to upload document')
      }
    },
  })

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!chatInput.trim() || isStreaming) return

    const message = chatInput.trim()
    setChatInput('')
    await sendChatMessage(message)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedFile) {
      setError('Please select a PDF file')
      return
    }

    try {
      setError('')
      await uploadDocument(selectedFile)
      setSelectedFile(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload document')
    }
  }

  return (
    <div className="w-full h-screen flex justify-center items-start pt-16">
      <div className="w-full max-w-[800px] px-4">
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex gap-2">
            <div className="flex-1">
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
              {selectedFile && (
                <p className="text-green-600 text-sm mt-1">Selected: {selectedFile.name}</p>
              )}
            </div>
            <button
              type="submit"
              disabled={!selectedFile}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Upload
            </button>
          </div>
        </form>

        <div className="border border-gray-200 rounded-lg p-4 bg-white">
          <h2 className="text-lg font-semibold mb-4">
            Ask Me About Your Document
          </h2>

          <div className="h-96 overflow-y-auto mb-4 p-4 bg-gray-50 rounded-lg">
            {messages.length === 0 ? (
              <p className="text-gray-500 text-center">
                Start a conversation...
              </p>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}
                >
                  <div
                    className={`inline-block max-w-[80%] p-3 rounded-lg ${
                      message.role === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-gray-800'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  </div>
                </div>
              ))
            )}
            {isStreaming && (
              <div className="text-left mb-4">
                <div className="inline-block bg-gray-200 text-gray-800 p-3 rounded-lg">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.1s' }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.2s' }}
                    ></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleChatSubmit} className="flex gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask me anything..."
              disabled={isStreaming}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
            />
            <button
              type="submit"
              disabled={isStreaming || !chatInput.trim()}
              className="px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
