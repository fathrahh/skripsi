import { Button, Card } from "@material-tailwind/react";
import { TiWarning } from "react-icons/ti";
import Navbar from "./components/Navbar";
import FeaturesForm from "./components/FeaturesForm";

function App() {
  return (
    <div className="container mx-auto">
      <Navbar />
      <h2 className="text-xl font-semibold">Klasifikasi Penyakit Stroke</h2>
      <Card
        variant="gradient"
        color="green"
        className="mt-4 p-4 flex flex-row gap-3 items-start"
      >
        <TiWarning size={28} />
        <div>
          <h2 className="text-xl font-bold">Attention!!!</h2>
          <p>This application is intended for education.</p>
        </div>
      </Card>
      <div className="mt-4">
        <FeaturesForm />
      </div>
    </div>
  );
}

export default App;
