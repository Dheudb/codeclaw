import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState, type MouseEvent } from "react";

function webpageURL(href?: string) {
  try {
    const url = new URL(href || "");
    return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

function fileReference(value: string) {
  return value.length <= 1000 && !/[\n\r<>]/.test(value) && /\.[a-z0-9]{1,12}$/i.test(value) &&
    (!/^[a-z][\w+.-]*:/i.test(value) || /^[a-z]:[\\/]/i.test(value));
}
function localLink(value?: string) {
  if (!value) return null;
  try { value = decodeURIComponent(value); } catch { return null; }
  return fileReference(value) ? value : null;
}
export default function Markdown({ text, onOpenFile }: { text: string; onOpenFile?: (path: string) => void }) {
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
      urlTransform={(url) => webpageURL(url) || (onOpenFile && fileReference(url) ? url : "")}
      components={{
        code: ({ children, className, node }) => {
          const path = String(children);
          return onOpenFile && !className && node?.position?.start.line === node?.position?.end.line && fileReference(path)
            ? <button className="inline-file-link" title={"查看文件 " + path} onClick={() => onOpenFile(path)}><code>{children}</code></button>
            : <code className={className}>{children}</code>;
        },
        a: ({ children, href }) => {
          const url = webpageURL(href);
          const path = localLink(href);
          return url ? <a href={url} title={url} className="markdown-link" target="_blank" rel="noopener noreferrer"
            onClick={(event) => openLink(event, url)}
            onAuxClick={(event) => { if (event.button === 1) openLink(event, url); }}>{children}</a>
            : onOpenFile && path ? <button className="inline-file-link" onClick={() => onOpenFile(path)}>{children}</button>
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
