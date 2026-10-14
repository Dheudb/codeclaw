import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState, type MouseEvent } from "react";

function webpageURL(href?: string) {
  try {
    const url = new URL(href || "");
    return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

export default function Markdown({ text }: { text: string }) {
  const [linkError, setLinkError] = useState("");
  function openLink(event: MouseEvent<HTMLAnchorElement>, url: string) {
    event.preventDefault();
    setLinkError("");
    window.codeclaw.invoke("external.open", { url }).catch(() => setLinkError("无法打开链接，请复制地址后手动打开：" + url));
  }
  return (
    <>
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ children, href }) => {
          const url = webpageURL(href);
          return url ? <a href={url} title={url} className="markdown-link" target="_blank" rel="noopener noreferrer"
            onClick={(event) => openLink(event, url)}
            onAuxClick={(event) => { if (event.button === 1) openLink(event, url); }}>{children}</a>
            : <span title={href}>{children}</span>;
        },
        img: ({ alt }) => <span className="muted">[图片：{alt}]</span>,
      }}
    >
      {text}
    </ReactMarkdown>
    {linkError && <p className="muted markdown-link-error" role="alert">{linkError}</p>}
    </>
  );
}
