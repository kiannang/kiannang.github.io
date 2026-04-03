import Link from "next/link";
import Image from "next/image";
import { ArrowUpRight, Pin } from "lucide-react";
import { getAllPosts } from "@/lib/posts";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";

export default function BlogPage() {
  const posts = getAllPosts();
  const pinnedPost = posts.find((p) => p.pinned);
  const otherPosts = posts.filter((p) => !p.pinned);

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
              {/* Featured pinned post */}
              {pinnedPost && (
                <Link
                  href={`/blog/${pinnedPost.slug}`}
                  className="group block overflow-hidden rounded-[28px] border border-line bg-accentSoft transition-all hover:-translate-y-0.5 hover:border-accent/40 hover:shadow-md"
                >
                  {pinnedPost.image && (
                    <div
                      className="relative w-full overflow-hidden"
                      style={{ aspectRatio: "21/9" }}
                    >
                      <Image
                        src={pinnedPost.image}
                        alt={pinnedPost.title}
                        fill
                        className="object-cover transition-transform duration-500 group-hover:scale-[1.02]"
                        style={{ objectPosition: pinnedPost.imagePosition ?? "center" }}
                        sizes="(max-width: 768px) 100vw, 896px"
                        priority
                      />
                    </div>
                  )}
                  <div className="p-6 sm:p-8">
                    <div className="mb-3 flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.2em] text-accent">
                      <Pin className="h-3 w-3" />
                      Pinned
                    </div>
                    <div className="flex items-start justify-between gap-4">
                      <div className="space-y-2">
                        <div className="flex flex-wrap gap-2">
                          {pinnedPost.tags.map((tag) => (
                            <span
                              key={tag}
                              className="rounded-full border border-line px-3 py-0.5 font-mono text-[11px] uppercase tracking-[0.2em] text-muted"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                        <h2 className="text-2xl font-bold text-foreground transition-colors group-hover:text-accent sm:text-3xl">
                          {pinnedPost.title}
                        </h2>
                        <p className="text-sm leading-7 text-muted">{pinnedPost.description}</p>
                        <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">
                          {new Date(pinnedPost.date).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "long",
                            day: "numeric",
                          })}
                        </p>
                      </div>
                      <ArrowUpRight className="mt-1 h-5 w-5 shrink-0 text-muted transition-colors group-hover:text-accent" />
                    </div>
                  </div>
                </Link>
              )}

              {/* Regular posts */}
              {otherPosts.map((post) => (
                <Link
                  key={post.slug}
                  href={`/blog/${post.slug}`}
                  className="group flex items-center gap-5 overflow-hidden rounded-[20px] border border-line bg-accentSoft px-6 py-5 transition-all hover:-translate-y-0.5 hover:border-accent/40 hover:shadow-md"
                >
                  <div className="min-w-0 flex-1 space-y-1.5">
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
                    <h2 className="text-xl font-bold text-foreground transition-colors group-hover:text-accent">
                      {post.title}
                    </h2>
                    <p className="line-clamp-2 text-sm leading-6 text-muted">{post.description}</p>
                    <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">
                      {new Date(post.date).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "long",
                        day: "numeric",
                      })}
                    </p>
                  </div>
                  {post.image && (
                    <div className="relative h-24 w-36 shrink-0 overflow-hidden rounded-[14px]">
                      <Image
                        src={post.image}
                        alt={post.title}
                        fill
                        className="object-cover transition-transform duration-500 group-hover:scale-[1.04]"
                        style={{ objectPosition: post.imagePosition ?? "center" }}
                        sizes="144px"
                      />
                    </div>
                  )}
                  <ArrowUpRight className="h-5 w-5 shrink-0 text-muted transition-colors group-hover:text-accent" />
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
