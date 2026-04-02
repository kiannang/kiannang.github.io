"use client";

import Link from "next/link";
import { Github, Linkedin, Menu, X } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { site } from "@/data/site";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "#home", label: "Home", id: "home" },
  { href: "#publications", label: "Publications", id: "publications" },
  { href: "#experience", label: "Experience", id: "experience" },
  { href: "#contact", label: "Contact", id: "contact" },
];

export function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState("home");
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    const sections = navItems
      .map((item) => document.getElementById(item.id))
      .filter((section): section is HTMLElement => Boolean(section));

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (visible?.target.id) {
          setActiveSection(visible.target.id);
        }
      },
      {
        rootMargin: "-28% 0px -50% 0px",
        threshold: [0.2, 0.45, 0.7],
      },
    );

    sections.forEach((section) => observer.observe(section));

    const onScroll = () => setScrolled(window.scrollY > 14);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    return () => {
      observer.disconnect();
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  const navClass = useMemo(
    () =>
      cn(
        "fixed inset-x-0 top-0 z-50 mx-auto mt-4 flex w-[calc(100%-1rem)] max-w-7xl items-center justify-between rounded-2xl border px-4 py-3 sm:w-[calc(100%-3rem)] sm:px-5",
        scrolled
          ? "border-line bg-panelStrong shadow-[0_14px_48px_rgba(0,0,0,0.35)] backdrop-blur-xl"
          : "border-transparent bg-transparent",
      ),
    [scrolled],
  );

  return (
    <header className={navClass}>
      <Link href="#home" className="group flex items-center gap-3">
        <span className="font-display italic text-2xl text-accent">Kianna Ng</span>
      </Link>

      <nav className="hidden items-center gap-1 md:flex">
        {navItems.map((item) => (
          <Link
            key={item.id}
            href={item.href}
            className={cn(
              "rounded-full px-4 py-2 text-sm text-muted hover:text-foreground",
              activeSection === item.id && "bg-white/5 text-foreground",
            )}
          >
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="hidden items-center gap-2 md:flex">
        <Link
          aria-label="GitHub"
          href={site.links.github}
          target="_blank"
          rel="noreferrer"
          className="rounded-full border border-line p-2 text-muted hover:border-accent/40 hover:text-foreground"
        >
          <Github className="h-4 w-4" />
        </Link>
        <Link
          aria-label="LinkedIn"
          href={site.links.linkedin}
          target="_blank"
          rel="noreferrer"
          className="rounded-full border border-line p-2 text-muted hover:border-accent/40 hover:text-foreground"
        >
          <Linkedin className="h-4 w-4" />
        </Link>
        <Link
          href="/resume/CV.pdf"
          className="rounded-full border border-accent/30 bg-accentSoft px-4 py-2 text-sm font-medium text-foreground hover:border-accent/60 hover:bg-white/10"
        >
          CV
        </Link>
      </div>

      <button
        type="button"
        aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
        aria-expanded={menuOpen}
        className="rounded-full border border-line p-2 text-muted md:hidden"
        onClick={() => setMenuOpen((value) => !value)}
      >
        {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>

      <AnimatePresence initial={false}>
        {menuOpen && (
          <motion.div
            initial={reduceMotion ? false : { opacity: 0, y: -12 }}
            animate={reduceMotion ? {} : { opacity: 1, y: 0 }}
            exit={reduceMotion ? {} : { opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="absolute left-0 right-0 top-[calc(100%+0.75rem)] rounded-3xl border border-line bg-panelStrong p-3 shadow-glow md:hidden"
          >
            <div className="flex flex-col gap-1">
              {navItems.map((item) => (
                <Link
                  key={item.id}
                  href={item.href}
                  onClick={() => setMenuOpen(false)}
                  className={cn(
                    "rounded-2xl px-4 py-3 text-sm text-muted hover:bg-white/5 hover:text-foreground",
                    activeSection === item.id && "bg-white/5 text-foreground",
                  )}
                >
                  {item.label}
                </Link>
              ))}
            </div>
            <div className="mt-3 flex items-center gap-2 border-t border-line pt-3">
              <Link
                href={site.links.github}
                target="_blank"
                rel="noreferrer"
                className="rounded-full border border-line p-2 text-muted"
                aria-label="GitHub"
              >
                <Github className="h-4 w-4" />
              </Link>
              <Link
                href={site.links.linkedin}
                target="_blank"
                rel="noreferrer"
                className="rounded-full border border-line p-2 text-muted"
                aria-label="LinkedIn"
              >
                <Linkedin className="h-4 w-4" />
              </Link>
              <Link href="/resume/CV.pdf" className="ml-auto rounded-full border border-accent/30 bg-accentSoft px-4 py-2 text-sm">
                CV
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
