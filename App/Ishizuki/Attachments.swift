// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What gets dropped on the composer: pictures the pack looks at, and files it is pointed at or
// handed outright.

import AppKit
import Foundation
import ImageIO
import IshizukiKit
import UniformTypeIdentifiers

/// One thing waiting to go out with the next turn.
struct Attachment: Identifiable, Equatable, Sendable {
  enum Kind: Equatable, Sendable {
    /// Seen by the vision tower rather than read as text.
    case image
    /// Read as text, either by pointing the agent at it or by handing it over whole.
    case text
    /// Neither: named so the model knows it exists, and nothing more.
    case opaque
  }

  var id = UUID()
  var url: URL
  var kind: Kind
  var byteCount: Int
  /// Set when the file was written by a paste rather than dragged in, so it can be swept up.
  var isTemporary = false

  var name: String { url.lastPathComponent }

  var icon: String {
    switch kind {
    case .image: "photo"
    case .text: "doc.text"
    case .opaque: "doc"
    }
  }

  /// What a file over this size is not: something to paste into a prompt. Beyond it a file is
  /// pointed at instead, whichever side of the workspace it is on.
  static let inlineByteCap = 48 * 1024

  static func make(from url: URL) -> Attachment? {
    let standardized = url.standardizedFileURL
    let values = try? standardized.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey])
    guard values?.isDirectory != true else { return nil }
    guard FileManager.default.isReadableFile(atPath: standardized.path) else { return nil }
    return Attachment(
      url: standardized,
      kind: kind(of: standardized),
      byteCount: values?.fileSize ?? 0)
  }

  private static func kind(of url: URL) -> Kind {
    let type = UTType(filenameExtension: url.pathExtension)
    if let type, type.conforms(to: .image) { return .image }
    if let type, type.conforms(to: .text) || type.conforms(to: .sourceCode) { return .text }
    if type == nil || type?.conforms(to: .data) == true {
      return looksTextual(url) ? .text : .opaque
    }
    return .opaque
  }

  /// A file with no useful type declared is judged by its first kilobyte: a NUL byte means it
  /// is not something to put in a prompt.
  private static func looksTextual(_ url: URL) -> Bool {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
    defer { try? handle.close() }
    guard let head = try? handle.read(upToCount: 1024), !head.isEmpty else { return false }
    return !head.contains(0)
  }
}

/// The drop's own reading of what it was handed, so the composer can say why something was
/// refused rather than quietly dropping it.
struct AttachmentIntake {
  var accepted: [Attachment] = []
  var refused: [String] = []

  static func read(_ urls: [URL]) -> AttachmentIntake {
    var intake = AttachmentIntake()
    for url in urls {
      if let attachment = Attachment.make(from: url) {
        intake.accepted.append(attachment)
      } else {
        intake.refused.append(url.lastPathComponent)
      }
    }
    return intake
  }

  /// An image on the pasteboard with no file behind it, written out so everything downstream
  /// can treat an attachment as a file and nothing more.
  static func readPasteboard() -> AttachmentIntake {
    let board = NSPasteboard.general
    if let urls = board.readObjects(forClasses: [NSURL.self]) as? [URL], !urls.isEmpty {
      return read(urls)
    }
    guard
      let type = board.availableType(from: [.png, .tiff]),
      let data = board.data(forType: type)
    else { return AttachmentIntake() }

    let png: Data? =
      type == .png
      ? data
      : NSBitmapImageRep(data: data)?.representation(using: .png, properties: [:])
    guard let png else { return AttachmentIntake() }

    let url = FileManager.default.temporaryDirectory
      .appending(path: "ishizuki-paste-\(UUID().uuidString.prefix(8)).png")
    guard (try? png.write(to: url)) != nil else { return AttachmentIntake() }

    var intake = AttachmentIntake()
    intake.accepted.append(
      Attachment(url: url, kind: .image, byteCount: png.count, isTemporary: true))
    return intake
  }
}

/// How a set of attachments reaches the model: the lines added to the prompt, and the pictures
/// handed to the vision tower alongside it.
struct AttachmentBundle {
  var preamble: String
  var images: [URL]

  /// Text inside the workspace is named rather than pasted — the agent has a read tool, and a
  /// file it can reach is better read than quoted at it. Everything else is handed over whole,
  /// up to the point where handing it over stops being reasonable.
  static func build(_ attachments: [Attachment], workspace: URL?) -> AttachmentBundle {
    var lines: [String] = []
    var images: [URL] = []

    for attachment in attachments {
      switch attachment.kind {
      case .image:
        images.append(attachment.url)
        lines.append("Attached image: \(attachment.name)")

      case .text:
        if let relative = relativePath(of: attachment.url, in: workspace) {
          lines.append("Attached file: \(relative)")
        } else if attachment.byteCount <= Attachment.inlineByteCap,
          let contents = try? String(contentsOf: attachment.url, encoding: .utf8)
        {
          lines.append(
            "Attached file \(attachment.name):\n```\n"
              + contents.trimmingCharacters(in: .newlines) + "\n```")
        } else {
          lines.append("Attached file: \(attachment.url.path) (outside the workspace)")
        }

      case .opaque:
        lines.append(
          "Attached file: \(attachment.url.path) "
            + "(\(ReadoutFormat.compact(attachment.byteCount)), not text)")
      }
    }

    return AttachmentBundle(preamble: lines.joined(separator: "\n\n"), images: images)
  }

  private static func relativePath(of url: URL, in workspace: URL?) -> String? {
    guard let workspace else { return nil }
    let root = workspace.resolvingSymlinksInPath().standardizedFileURL.path
    let path = url.resolvingSymlinksInPath().standardizedFileURL.path
    guard path.hasPrefix(root + "/") else { return nil }
    return String(path.dropFirst(root.count + 1))
  }
}

/// A small picture of a picture, for the chip over the composer. Made through ImageIO so a
/// twelve-megapixel screenshot is never decoded in full just to be drawn at twenty-two points.
enum Thumbnail {
  static func png(of url: URL, maxPixel: Int = 96) -> Data? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: maxPixel,
    ]
    guard
      let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    else { return nil }
    return NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
  }
}
