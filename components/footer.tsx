"use client";

import Link from "next/link";
import { site } from "@/data/site";

const elsewhere = [
  { label: "GitHub", href: site.links.github },
  { label: "LinkedIn", href: site.links.linkedin },
];

export function Footer() {
  return (
    <footer className="mx-auto max-w-[800px] px-4 pb-6 pt-6 text-center text-[11px] leading-5">
      <p>
        Find my full <Link href="/resume/CV.pdf" className="classic-link">Curriculum Vitae</Link> here.
      </p>

      <div className="mt-5">
        <label htmlFor="elsewhere">Visit me elsewhere around the web:</label>
        <div className="mt-1">
          <select
            id="elsewhere"
            defaultValue=""
            onChange={(event) => {
              if (event.target.value) window.open(event.target.value, "_blank", "noreferrer");
              event.target.value = "";
            }}
            className="w-[180px] rounded-[4px] border border-[#7a7a7a] bg-white px-1 py-[2px] text-[12px]"
          >
            <option value="" disabled>
              Choose...
            </option>
            {elsewhere.map((item) => (
              <option key={item.href} value={item.href}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <p className="mt-5">Copyright &copy; {new Date().getFullYear()} Kianna Ng. All rights reserved.</p>
      <p>Thanks for visiting.</p>

      <p className="mx-auto mt-4 max-w-[400px] bg-[radial-gradient(ellipse_at_center,rgba(255,255,255,0.75),transparent_70%)] py-2 text-[12px] font-bold">
        Powered by Next.js on GitHub Pages
      </p>
    </footer>
  );
}
