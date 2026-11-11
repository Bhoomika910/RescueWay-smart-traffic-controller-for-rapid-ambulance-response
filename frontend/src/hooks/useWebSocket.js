// src/hooks/useWebSocket.js
import { useEffect, useRef, useState } from "react";

export default function useWebSocket(url) {
  const [lastEvent, setLastEvent]   = useState(null);
  const [connected, setConnected]   = useState(false);
  const ws = useRef(null);
  const reconnectTimeout = useRef(null);

  useEffect(() => {
    function connect() {
      if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
        try {
          ws.current.close();
        } catch (e) {
          console.error("Error closing WebSocket:", e);
        }
      }

      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        setConnected(true);
        console.log("WebSocket connected");
        if (reconnectTimeout.current) {
          clearTimeout(reconnectTimeout.current);
          reconnectTimeout.current = null;
        }
      };

      ws.current.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          setLastEvent(data);
        } catch (err) {
          console.error("WS parse error:", err);
        }
      };

      ws.current.onclose = () => {
        setConnected(false);
        console.log("WebSocket disconnected — retrying in 2s");
        reconnectTimeout.current = setTimeout(connect, 2000);
      };

      ws.current.onerror = (err) => {
        console.error("WebSocket error:", err);
        if (ws.current) {
          ws.current.close();
        }
      };
    }

    connect();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [url]);

  return { lastEvent, connected };
}
