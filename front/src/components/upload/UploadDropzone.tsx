"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";

const ACCEPTED = {
  "application/pdf": [".pdf"],
  "text/plain": [".txt"],
  "image/png": [".png"],
  "image/jpeg": [".jpg", ".jpeg"],
};

interface UploadDropzoneProps {
  onFilesSelected: (files: File[]) => void;
  activeSessionId?: string;
}

export function UploadDropzone({ onFilesSelected, activeSessionId }: UploadDropzoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) {
        onFilesSelected(accepted);
      }
    },
    [onFilesSelected],
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } =
    useDropzone({
      onDrop,
      accept: ACCEPTED,
      maxSize: 50 * 1024 * 1024, // 50 MB
    });

  return (
    <div>
      <div
        {...getRootProps()}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
          isDragActive
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/50 hover:bg-muted/50"
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto mb-3 size-8 text-muted-foreground" />
        <p className="font-medium text-foreground text-sm">
          {isDragActive
            ? "Soltá los archivos acá"
            : "Arrastrá archivos acá o hacé clic para seleccionar"}
        </p>
        <p className="mt-1 text-muted-foreground text-xs">
          PDF, TXT, PNG o JPG (máx. 50 MB)
        </p>
      </div>

      {fileRejections.length > 0 && (
        <div className="mt-2 rounded-md bg-red-50 px-3 py-2 text-red-700 text-xs dark:bg-red-950 dark:text-red-300">
          {fileRejections.map(({ file, errors }, i) => (
            <p key={i}>
              <strong>{file.name}</strong>:{" "}
              {errors.map((e) => e.message).join(", ")}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
