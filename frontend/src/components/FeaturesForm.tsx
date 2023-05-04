import { Button, Input, Option, Select } from "@material-tailwind/react";
import { FormEvent, useRef, useState } from "react";

export default function FeaturesForm() {
  const [locate, setLocate] = useState("");

  const onSubmitForm = (ev: FormEvent<HTMLFormElement>): void => {
    ev.preventDefault();
    const formData = new FormData(ev.currentTarget);
    console.log(Object.fromEntries(formData));
  };

  return (
    <form onSubmit={onSubmitForm}>
      <div className="grid grid-cols-2 gap-6">
        <Input name="smoker" className="col-span-1" size="md" label="Smoking" />
        <Input className="col-span-1" size="md" label="Smoking" />
        <Input className="col-span-1" size="md" label="Smoking" />
        <Input className="col-span-1" size="md" label="Smoking" />
        <Input className="col-span-1" size="md" label="Smoking" />
        <Select
          value={locate}
          onChange={(value) => setLocate(value ?? "")}
          name="lokasiTinggal"
          label="Lokasi Tinggal"
        >
          <Option value="resident">Resident</Option>
          <Option value="pedalaman">Pedalaman</Option>
          <Option value="Perdamaian">Perdamaian</Option>
        </Select>
      </div>
      <Button type="submit" variant="gradient" color="orange">
        Predict
      </Button>
    </form>
  );
}
