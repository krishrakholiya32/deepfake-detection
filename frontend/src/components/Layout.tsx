import { NavLink, Outlet } from 'react-router-dom'
import { ScanFace, History } from 'lucide-react'

export default function Layout() {
  return (
    <div className="flex min-h-screen bg-df-bg text-white">
      <aside className="w-56 border-r border-df-border bg-df-surface p-4">
        <h1 className="mb-8 text-lg font-bold">Deepfake Detection</h1>
        <nav className="flex flex-col gap-2">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `flex items-center gap-2 rounded-md px-3 py-2 ${isActive ? 'bg-df-accent/20 text-df-accent' : 'text-df-muted'}`
            }
          >
            <ScanFace size={18} /> Detect
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) =>
              `flex items-center gap-2 rounded-md px-3 py-2 ${isActive ? 'bg-df-accent/20 text-df-accent' : 'text-df-muted'}`
            }
          >
            <History size={18} /> History
          </NavLink>
        </nav>
      </aside>
      <main className="flex-1 p-8">
        <Outlet />
      </main>
    </div>
  )
}
