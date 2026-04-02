import Link from "next/link";
import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { getAllPosts } from "@/lib/posts";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";

export default function BlogPage() {
  const posts = getAllPosts();

  return (
    <div className="relative min-h-screen">
      <Navbar />

      <main className="mx-auto w-full max-w-4xl px-4 pb-20 pt-32 sm:px-6 lg:px-8">
        <div className="space-y-12">
          <div className="space-y-3">
            <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent">/ blog</p>
            <h1 className="leading-none">
              <span className="block text-5xl font-black text-foreground sm:text-6xl">Writing &</span>
              <span className="block font-display italic text-6xl text-foreground sm:text-7xl">Thinking.</span>
            </h1>
            <p className="max-w-xl text-lg text-muted">Notes on research, ideas, and whatever else is on my mind.</p>
          </div>

          {posts.length === 0 ? (
            <p className="text-muted">No posts yet — check back soon.</p>
          ) : (
            <div className="space-y-4">
              {posts.map((post) => (
                <Link
                  key={post.slug}
                  href={`/blog/${post.slug}`}
                  className="group flex items-start justify-between gap-6 rounded-[28px] border border-line bg-accentSoft p-6 transition-all hover:border-accent/40"
                >
                  {post.image && (
                    <div className="relative mb-4 aspect-[16/9] w-full overflow-hidden rounded-[20px]">
                      <Image
                        src={post.image}
                        alt={post.title}
                        fill
                        className="object-cover"
                        sizes="(max-width: 768px) 100vw, 672px"
                      />
                    </div>
                  )}
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-2">
                      <div className="flex flex-wrap gap-2">
                        {post.tags.map((tag) => (
                          <span
                            key={tag}
                            className="rounded-full border border-line px-3 py-0.5 font-mono text-[11px] uppercase tracking-[0.2em] text-muted"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      <h2 className="text-2xl font-bold text-foreground transition-colors group-hover:text-accent">
                        {post.title}
                      </h2>
                      <p className="text-sm leading-7 text-muted">{post.description}</p>
                      <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">
                        {new Date(post.date).toLocaleDateString("en-US", {
                          year: "numeric",
                          month: "long",
                          day: "numeric",
                        })}
                      </p>
                    </div>
                    <ArrowUpRight className="mt-1 h-5 w-5 shrink-0 text-muted transition-colors group-hover:text-accent" />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
