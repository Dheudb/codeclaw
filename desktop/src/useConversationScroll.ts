import { useEffect, useLayoutEffect, useRef } from "react";

// Follow streaming output only while the reader is at the end of the conversation.
export default function useConversationScroll(sessionId: string, messages: unknown, interactions: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const displayed = useRef("");
  const positions = useRef(new Map<string, { top: number; following: boolean }>());
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    if (displayed.current !== sessionId) {
      displayed.current = sessionId;
      const saved = positions.current.get(sessionId);
      following.current = saved?.following ?? true;
      element.scrollTop = following.current ? element.scrollHeight : saved?.top ?? 0;
    } else if (following.current) {
      // An immediate container scroll avoids overlapping smooth-scroll animations
      // and never scrolls an ancestor or a neighboring pane.
      element.scrollTop = element.scrollHeight;
    }
  }, [sessionId, messages, interactions]);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      if (following.current) element.scrollTop = element.scrollHeight;
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return {
    ref,
    followLatest: () => { following.current = true; },
    onWheel: (event: React.WheelEvent<HTMLDivElement>) => {
      if (event.deltaY < 0) following.current = false;
    },
    onScroll: () => {
      const element = ref.current;
      if (!element) return;
      following.current = element.scrollHeight - element.clientHeight - element.scrollTop <= 2;
      positions.current.set(displayed.current, { top: element.scrollTop, following: following.current });
    },
  };
}
