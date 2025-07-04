import { Link } from '@tanstack/react-router'
import logo from '/smith.png'

export default function Header() {
  return (
    <header className="p-2 flex gap-2 bg-white text-black justify-between items-center">
      <img src={logo} className="w-12 h-12" />
      <nav className="flex flex-row">
        <div className="px-2 font-bold text-xl">
          <Link to="/">Home</Link>
        </div>
      </nav>
      <nav className="flex flex-row">
        <div className="px-2 text-xl">
          <Link to="/login">Login</Link>
        </div>
      </nav>
    </header>
  )
}
