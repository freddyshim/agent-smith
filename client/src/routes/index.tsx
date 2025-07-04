import { createFileRoute } from '@tanstack/react-router'
import {
  useQuery,
  queryOptions,
  experimental_streamedQuery as streamedQuery,
} from '@tanstack/react-query'

export const Route = createFileRoute('/')({
  component: App,
})

function App() {
  const chatQueryOptions = queryOptions({
    queryKey: ['chat'],
    queryFn: streamedQuery({
      queryFn: () => {
        return {
          async *[Symbol.asyncIterator]() {
            const res = await fetch('http://localhost:8000/chat', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                prompt: 'How are you doing today?',
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
                yield decodedChunk
              }
            } else {
              throw new Error('failed to fetch data')
            }
          },
        }
      },
    }),
  })

  const { data = [] } = useQuery(chatQueryOptions)

  return (
    <div className="w-full h-screen flex justify-center items-start">
      <p className="w-full max-w-[800px]">{data.join('')}</p>
    </div>
  )
}
