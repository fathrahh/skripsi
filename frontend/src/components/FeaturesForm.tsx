import { Button, Input } from "@material-tailwind/react";
import type { FormEvent } from "react";

export default function FeaturesForm() {
  const onSubmitForm = (ev: FormEvent<HTMLFormElement>): void => {
    ev.preventDefault();
  };

  return (
    <form onSubmit={onSubmitForm}>
      <div>
        
        <Input />
      </div>
      <Button variant="gradient" color="orange">
        Predict
      </Button>
    </form>
  );
}
