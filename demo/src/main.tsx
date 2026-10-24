import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { initTheme, TooltipProvider } from "@vitavision/ui";
import App from "./App";
import "./index.css";

initTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <TooltipProvider>
      <App />
    </TooltipProvider>
  </StrictMode>,
);
