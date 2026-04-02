export function Footer() {
  return (
    <footer className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-8 text-sm text-muted sm:px-6 lg:px-8">
      <p>Kianna Ng</p>
      <p>{new Date().getFullYear()}</p>
      <p>Thanks for visiting.</p>
    </footer>
  );
}
