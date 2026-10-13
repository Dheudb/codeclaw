import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
export default function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ children, href }) => (
          <span title={href} className="markdown-link">
            {children}
          </span>
        ),
        img: ({ alt }) => <span className="muted">[图片：{alt}]</span>,
      }}
    >
      {text}
    </ReactMarkdown>
  );
}
