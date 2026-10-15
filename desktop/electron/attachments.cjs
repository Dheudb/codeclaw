const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { atomicJSON } = require("./catalog.cjs");
const IMAGE_TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp" };
const MAX_FILE = 64 * 1024 * 1024, MAX_IMAGE = 20 * 1024 * 1024;

class Attachments {
  constructor(catalog, nativeImage) { this.catalog = catalog; this.nativeImage = nativeImage; }
  directory(row) {
    const directory = path.join(this.catalog.artifacts(row.project), row.sessionId, "attachments");
    const base = fs.realpathSync(this.catalog.dataRoot), relative = path.relative(base, directory);
    if (relative.startsWith("..") || path.isAbsolute(relative)) throw new Error("附件目录无效");
    let probe = base;
    for (const part of relative.split(path.sep)) {
      probe = path.join(probe, part);
      if (fs.existsSync(probe) && fs.lstatSync(probe).isSymbolicLink()) throw new Error("附件目录不能是链接");
    }
    fs.mkdirSync(directory, { recursive: true });
    return directory;
  }
  get(row, id) {
    if (typeof id !== "string" || !/^[a-f0-9-]{36}$/.test(id)) throw new Error("附件 ID 无效");
    const directory = this.directory(row), target = path.join(directory, id + ".json");
    if (fs.lstatSync(target).isSymbolicLink()) throw new Error("附件记录无效");
    const item = JSON.parse(fs.readFileSync(target, "utf8"));
    if (item.id !== id || path.basename(item.file) !== item.file || !item.file.startsWith(id + ".")) throw new Error("附件记录无效");
    const file = path.join(directory, item.file);
    if (fs.lstatSync(file).isSymbolicLink() || !fs.statSync(file).isFile()) throw new Error("附件已丢失");
    return { id, name: item.name, size: item.size, kind: item.kind, preview: item.preview || "", path: file };
  }
  list(row, ids = []) {
    if (!Array.isArray(ids) || ids.length > 10) throw new Error("每条消息最多添加 10 个附件");
    return [...new Set(ids)].map((id) => this.get(row, id));
  }
  async import(row, files, existing = []) {
    const retained = this.list(row, existing);
    if (!Array.isArray(files) || !files.length || retained.length + files.length > 10) throw new Error("每条消息最多添加 10 个附件");
    const directory = this.directory(row), created = [], result = [];
    try {
      for (const input of files) {
        const id = crypto.randomUUID();
        let bytes, original, name, size;
        if (input.path) {
          original = await fs.promises.realpath(input.path);
          const stat = await fs.promises.stat(original);
          if (!stat.isFile()) throw new Error("请添加文件，暂不支持文件夹");
          name = path.basename(original); size = stat.size;
        } else {
          if (typeof input.data !== "string" || input.data.length > Math.ceil(MAX_IMAGE * 4 / 3) + 100) throw new Error("粘贴的图片过大（最大 20 MB）");
          const match = /^data:image\/(png|jpeg|webp|gif|bmp);base64,([A-Za-z0-9+/]+={0,2})$/.exec(input.data);
          if (!match) throw new Error("粘贴的图片格式无效");
          bytes = Buffer.from(match[2], "base64"); size = bytes.length;
          name = "粘贴图片-" + id.slice(0, 8) + "." + (match[1] === "jpeg" ? "jpg" : match[1]);
        }
        const extension = path.extname(name).toLowerCase(), kind = IMAGE_TYPES[extension] ? "image" : "file";
        if (size > (kind === "image" ? MAX_IMAGE : MAX_FILE)) throw new Error(name + (kind === "image" ? " 超过图片上限 20 MB" : " 超过文件上限 64 MB"));
        const file = id + (extension && /^[.a-z0-9_-]{1,20}$/.test(extension) ? extension : ".bin");
        const destination = path.join(directory, file); created.push(destination);
        if (bytes) await fs.promises.writeFile(destination, bytes, { flag: "wx" });
        else await fs.promises.copyFile(original, destination, fs.constants.COPYFILE_EXCL);
        // Enforce the size limit again in case a source file changed during copying.
        size = (await fs.promises.stat(destination)).size;
        if (size > (kind === "image" ? MAX_IMAGE : MAX_FILE)) throw new Error("附件在读取时变大，请重新添加");
        let preview = "";
        if (kind === "image" && this.nativeImage) {
          const image = this.nativeImage.createFromPath(destination);
          if (image.isEmpty()) throw new Error(name + " 不是可读取的图片");
          const dimensions = image.getSize(), scale = Math.min(112 / dimensions.width, 80 / dimensions.height, 1);
          preview = image.resize({ width: Math.max(1, Math.round(dimensions.width * scale)), height: Math.max(1, Math.round(dimensions.height * scale)), quality: "good" }).toDataURL();
        }
        const metadata = path.join(directory, id + ".json"); created.push(metadata);
        atomicJSON(metadata, { id, file, name, size, kind, preview });
        result.push(this.get(row, id));
      }
      return [...retained, ...result];
    } catch (error) {
      for (const file of created.reverse()) if (fs.existsSync(file)) fs.unlinkSync(file);
      throw error;
    }
  }
}
module.exports = { Attachments };
