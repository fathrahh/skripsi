import Image from "next/image";
import { SignOutButton } from "@clerk/nextjs";

export default function Navbar() {
  return (
    <nav className="py-6 flex items-center justify-between ">
      <Image
        width={48}
        height={48}
        className="w-12 h-12"
        src="/logo_upnvj.png"
        alt="logo-upn"
      />

      <SignOutButton />
    </nav>
  );
}
