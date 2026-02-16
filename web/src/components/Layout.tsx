import type { ReactNode } from "react";
import Sidebar from "./Sidebar";
import StatusBar from "./StatusBar";

interface Props {
  children: ReactNode;
}

export default function Layout({ children }: Props) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto p-4 md:p-6">{children}</main>
        <StatusBar />
      </div>
    </div>
  );
}
