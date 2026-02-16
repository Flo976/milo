import { Routes, Route, Navigate } from "react-router-dom";
import ConversationScreen from "./features/conversation/ConversationScreen";
import TranscriptionScreen from "./features/transcription/TranscriptionScreen";
import TtsScreen from "./features/tts/TtsScreen";
import AdminDashboard from "./features/admin/AdminDashboard";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<ConversationScreen />} />
      <Route path="/transcription" element={<TranscriptionScreen />} />
      <Route path="/tts" element={<TtsScreen />} />
      <Route path="/admin" element={<AdminDashboard />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
