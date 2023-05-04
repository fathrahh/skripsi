import upnLogo from "./../assets/logo_upnvj.png";

export default function Navbar() {
  return (
    <nav className="py-6 flex items-center">
      <img className="w-12 h-12" src={upnLogo} />
      <ul className="flex gap-2 ml-6">
        <li>Home</li>
        <li>Predict</li>
      </ul>
    </nav>
  );
}
