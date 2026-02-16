import { useEffect, useState } from "react";
import { api } from "../api/endpoints";
import type { HealthResponse } from "../api/types";

export function useHealth(intervalMs = 30000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const fetch = async () => {
      try {
        const data = await api.health();
        if (active) {
          setHealth(data);
          setError(null);
        }
      } catch (err) {
        if (active) setError(String(err));
      }
    };

    void fetch();
    const id = setInterval(fetch, intervalMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { health, error };
}
