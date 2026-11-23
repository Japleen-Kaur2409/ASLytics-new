// Draws one pose frame as a 3D stick figure. A stand-in for the VRM avatar:
// same scene, camera and data flow, so swapping the figure for an avatar later
// only touches this file.

import * as THREE from "three";
import { BODY_BONE_COUNT, BONES, FACE, FRAME_SIZE, NUM_POINTS, type Sequence } from "./landmarks";

const UP = new THREE.Vector3(0, 1, 0);

export class Renderer {
  readonly canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.OrthographicCamera;
  private bones: THREE.InstancedMesh;
  private joints: THREE.InstancedMesh;
  private face: THREE.Points;
  private head: THREE.Mesh;
  private points = Array.from({ length: NUM_POINTS }, () => new THREE.Vector3());
  private matrix = new THREE.Matrix4();
  private quat = new THREE.Quaternion();
  private scale = new THREE.Vector3();
  private mid = new THREE.Vector3();
  private dir = new THREE.Vector3();

  constructor(width: number, height: number) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(width, height);
    this.canvas = this.renderer.domElement;

    // Poses are in shoulder widths, centred between the shoulders.
    const halfH = 1.9, halfW = (halfH * width) / height;
    this.camera = new THREE.OrthographicCamera(-halfW, halfW, halfH, -halfH, 0.1, 20);
    this.camera.position.set(0, 0.15, 6);

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1, 2, 4);
    this.scene.add(key);

    const boneMaterial = new THREE.MeshStandardMaterial({ roughness: 0.6 });
    this.bones = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10), boneMaterial, BONES.length);
    const bodyColour = new THREE.Color(0x4b6584), handColour = new THREE.Color(0xf5c9a6);
    BONES.forEach((_, i) => this.bones.setColorAt(i, i < BODY_BONE_COUNT ? bodyColour : handColour));
    this.scene.add(this.bones);

    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 8), boneMaterial, FACE);
    for (let i = 0; i < FACE; i++) this.joints.setColorAt(i, i < 8 ? bodyColour : handColour);
    this.scene.add(this.joints);

    this.head = new THREE.Mesh(
      new THREE.SphereGeometry(1, 24, 16),
      new THREE.MeshStandardMaterial({ color: 0xf5c9a6, roughness: 0.8 }),
    );
    this.scene.add(this.head);

    const faceGeometry = new THREE.BufferGeometry();
    faceGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array((NUM_POINTS - FACE) * 3), 3));
    this.face = new THREE.Points(faceGeometry, new THREE.PointsMaterial({ color: 0x2d3436, size: 3, sizeAttenuation: false }));
    this.scene.add(this.face);
  }

  /** Draw the pose at a fractional frame position, blending neighbouring frames. */
  draw(seq: Sequence, position: number): void {
    const clamped = Math.min(Math.max(position, 0), seq.n - 1);
    const t0 = Math.floor(clamped), t1 = Math.min(seq.n - 1, t0 + 1), w = clamped - t0;
    const a = t0 * FRAME_SIZE, b = t1 * FRAME_SIZE;
    for (let p = 0; p < NUM_POINTS; p++) {
      const k = p * 3;
      // Data: y down, z towards the camera is negative. Three.js: y up, z towards the camera positive.
      this.points[p].set(
        seq.frames[a + k] * (1 - w) + seq.frames[b + k] * w,
        -(seq.frames[a + k + 1] * (1 - w) + seq.frames[b + k + 1] * w),
        -(seq.frames[a + k + 2] * (1 - w) + seq.frames[b + k + 2] * w),
      );
    }

    BONES.forEach(([from, to], i) => {
      const p = this.points[from], q = this.points[to];
      const radius = i < BODY_BONE_COUNT ? 0.06 : 0.022;
      this.dir.subVectors(q, p);
      const length = this.dir.length();
      this.mid.addVectors(p, q).multiplyScalar(0.5);
      this.quat.setFromUnitVectors(UP, length > 1e-6 ? this.dir.divideScalar(length) : UP);
      this.matrix.compose(this.mid, this.quat, this.scale.set(radius, Math.max(length, 1e-4), radius));
      this.bones.setMatrixAt(i, this.matrix);
    });
    this.bones.instanceMatrix.needsUpdate = true;

    this.quat.identity();
    for (let p = 0; p < FACE; p++) {
      const radius = p < 8 ? 0.07 : 0.028;
      this.matrix.compose(this.points[p], this.quat, this.scale.set(radius, radius, radius));
      this.joints.setMatrixAt(p, this.matrix);
    }
    this.joints.instanceMatrix.needsUpdate = true;

    // Head: an ellipsoid behind the face points, sized from their spread.
    const facePositions = this.face.geometry.getAttribute("position") as THREE.BufferAttribute;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    this.mid.set(0, 0, 0);
    for (let p = FACE; p < NUM_POINTS; p++) {
      const v = this.points[p];
      facePositions.setXYZ(p - FACE, v.x, v.y, v.z + 0.5);
      this.mid.add(v);
      minX = Math.min(minX, v.x); maxX = Math.max(maxX, v.x);
      minY = Math.min(minY, v.y); maxY = Math.max(maxY, v.y);
    }
    facePositions.needsUpdate = true;
    this.mid.multiplyScalar(1 / (NUM_POINTS - FACE));
    const rx = Math.max(0.2, (maxX - minX) * 0.72), ry = Math.max(0.25, (maxY - minY) * 0.75);
    this.head.position.set(this.mid.x, this.mid.y + ry * 0.12, this.mid.z - rx);
    this.head.scale.set(rx, ry, rx);

    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.renderer.dispose();
    this.canvas.remove();
  }
}
