"""
Simulación urbana de tráfico con agentes Mesa y visualización pygame.

Instalación:
    ./.venv/Scripts/python.exe -m pip install "mesa[network]>=3,<4" pygame

Ejecución:
    ./.venv/Scripts/python.exe main.py

La cuadrícula y el modelo son 3x3 por defecto. Cada semáforo mide su propia
cola y toma sus propias decisiones. TrafficModel entrega mensajes entre
agentes, pero no decide los estados ni los tiempos de los semáforos.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import mesa
import pygame


# ---------------------------------------------------------------------------
# Configuración visual y del mapa
# ---------------------------------------------------------------------------

CITY_WIDTH = 700
CITY_HEIGHT = 600
PANEL_WIDTH = 360
PANEL_HEIGHT = 646
HEADER_HEIGHT = 54
FOOTER_HEIGHT = 50
WINDOW_SIZE = (CITY_WIDTH + PANEL_WIDTH, HEADER_HEIGHT + PANEL_HEIGHT)
DISPLAY_SIZE = (1600, 1000)
DISPLAY_MAP_SIZE = (1080, 936)
DISPLAY_PANEL_SIZE = (520, 936)
DISPLAY_HEADER_HEIGHT = 64
FPS = 150

ROAD_WIDTH = 48
ROAD_MARGIN_X = 98
ROAD_MARGIN_Y = 68

# Ejes de circulación. EW = este-oeste; NS = norte-sur.
EW = "EW"
NS = "NS"

GREEN = "VERDE"
YELLOW = "AMARILLO"
RED = "ROJO"

ASPHALT = (57, 67, 77)
GRASS = (211, 221, 222)
ROAD_MARKING = (211, 216, 208)
PANEL_BG = (17, 27, 39)
TEXT_COLOR = (236, 240, 244)
MUTED_TEXT = (165, 175, 186)
GREEN_COLOR = (47, 210, 105)
YELLOW_COLOR = (247, 199, 55)
RED_COLOR = (235, 70, 74)

VEHICLE_COLORS = [
    (117, 170, 190),
    (177, 197, 202),
    (190, 169, 133),
    (137, 160, 172),
    (176, 157, 180),
]


# ---------------------------------------------------------------------------
# Agente semáforo
# ---------------------------------------------------------------------------

class TrafficLightAgent(mesa.Agent):
    """Semáforo autónomo para una intersección."""

    GREEN_SECONDS = 6.0
    YELLOW_SECONDS = 1.4
    ALL_RED_SECONDS = 0.5
    MAX_GREEN_SECONDS = 12.0

    QUEUE_RANGE = 88.0
    ALERT_THRESHOLD = 2
    ALERT_COOLDOWN = 1.8
    MAX_EXTENSION_PER_ALERT = 2.5
    MAX_PENDING_EXTENSION = 4.0
    MAX_RECEIVING_QUEUE = 5

    def __init__(self, model: "TrafficModel", row: int, col: int):
        # En Mesa 3, unique_id se asigna automáticamente.
        super().__init__(model)

        self.row = row
        self.col = col
        self.x = model.road_x[col]
        self.y = model.road_y[row]

        self.green_axis = EW if (row + col) % 2 == 0 else NS
        self.state = GREEN
        self.phase_remaining = self.GREEN_SECONDS
        self.last_axis = self.green_axis

        self.queue_counts = {EW: 0, NS: 0}
        self.urgency = 0
        self.inbox: list[dict] = []
        self.pending_extension = {EW: 0.0, NS: 0.0}

        self.last_alert_time = -100.0
        self.last_decision = "Ciclo inicial"

    def state_for(self, axis: str) -> str:
        """Devuelve el color que ve un vehículo de este eje."""
        if self.state == RED or axis != self.green_axis:
            return RED
        return self.state

    def can_pass(self, axis: str) -> bool:
        """Permite cruzar únicamente con verde en el eje correspondiente."""
        return self.state == GREEN and self.green_axis == axis

    def receive_message(self, message: dict) -> None:
        """Recibe un mensaje directamente de otro agente semáforo."""
        self.inbox.append(message)

    def sense_local_traffic(self) -> None:
        """
        Calcula la métrica local de urgencia a partir de los vehículos que
        se aproximan a esta intersección por cada eje.
        """
        self.queue_counts[EW] = self.model.count_approaching(self, EW)
        self.queue_counts[NS] = self.model.count_approaching(self, NS)
        self.urgency = self.queue_counts[EW] + self.queue_counts[NS]

    def _neighbor_positions(self) -> list[tuple[int, int]]:
        """Intersecciones contiguas conectadas por el eje que está en verde."""
        if self.green_axis == EW:
            return [
                (self.row, self.col - 1),
                (self.row, self.col + 1),
            ]
        return [
            (self.row - 1, self.col),
            (self.row + 1, self.col),
        ]

    def maybe_broadcast_flow_alert(self) -> None:
        """
        Protocolo vecinal: si el semáforo va a liberar una cola grande,
        envía un Mensaje de Alerta de Flujo a los vecinos de esa vía.

        El agente vecino recibe el mensaje en su buzón y decide por sí mismo
        si extiende su verde. Este método solo hace el envío.
        """
        if not self.model.adaptive or self.state != GREEN:
            return
        if self.queue_counts[self.green_axis] < self.ALERT_THRESHOLD:
            return
        if self.model.simulation_time - self.last_alert_time < self.ALERT_COOLDOWN:
            return

        sent = False
        for position in self._neighbor_positions():
            neighbor = self.model.signals_by_position.get(position)
            if neighbor is None:
                continue

            message = {
                "type": "FLOW_ALERT",
                "sender": self.unique_id,
                "source": (self.row, self.col),
                "axis": self.green_axis,
                "volume": self.queue_counts[self.green_axis],
                "sent_at": self.model.simulation_time,
            }

            # Entrega entre agentes; no pasa por un controlador de semáforos.
            neighbor.receive_message(message)
            self.model.alerts_sent += 1
            self.model.register_flow_alert(self, neighbor)
            self.model.last_event = (
                f"Alerta de flujo: ({self.row + 1},{self.col + 1})"
                f" -> ({neighbor.row + 1},{neighbor.col + 1})"
            )
            sent = True

        if sent:
            self.last_alert_time = self.model.simulation_time

    def _process_messages(self) -> None:
        """
        Decisión autónoma del receptor. Si el flujo entrante coincide con el
        verde actual y su cola local permite recibir más vehículos, extiende
        ese verde. Si el eje aún no está activo, guarda una extensión limitada
        para cuando le toque el turno.
        """
        messages = self.inbox
        self.inbox = []

        for message in messages:
            axis = message["axis"]
            volume = int(message["volume"])

            # Un agente con una cola de recepción ya grande evita agravarla.
            if self.queue_counts[axis] > self.MAX_RECEIVING_QUEUE:
                self.last_decision = (
                    f"Alerta rechazada: cola {axis} saturada"
                )
                continue

            extension = min(
                self.MAX_EXTENSION_PER_ALERT,
                max(0.5, volume * 0.45),
            )

            if self.state == GREEN and self.green_axis == axis:
                self.phase_remaining = min(
                    self.MAX_GREEN_SECONDS,
                    self.phase_remaining + extension,
                )
                self.last_decision = (
                    f"Extiende verde {axis} +{extension:.1f}s"
                )
            else:
                self.pending_extension[axis] = min(
                    self.MAX_PENDING_EXTENSION,
                    self.pending_extension[axis] + extension,
                )
                self.last_decision = (
                    f"Reserva +{extension:.1f}s para {axis}"
                )

            self.model.last_event = (
                f"Semáforo ({self.row + 1},{self.col + 1}) "
                f"procesó alerta {axis}"
            )

    def _choose_next_axis(self) -> str:
        """
        Selección local de fase: compara las colas de entrada y usa el eje
        opuesto como desempate para repartir el servicio cuando la demanda es
        igual.
        """
        scores = {
            EW: self.queue_counts[EW] + self.pending_extension[EW] * 0.25,
            NS: self.queue_counts[NS] + self.pending_extension[NS] * 0.25,
        }

        if abs(scores[EW] - scores[NS]) < 0.01:
            return NS if self.last_axis == EW else EW
        return EW if scores[EW] > scores[NS] else NS

    def update_phase(self, dt: float) -> None:
        if self.model.adaptive:
            self._process_messages()
        else:
            self.inbox.clear()
        self.phase_remaining -= dt

        if self.phase_remaining > 0:
            return

        if self.state == GREEN:
            self.state = YELLOW
            self.phase_remaining = self.YELLOW_SECONDS
            self.last_decision = f"Amarillo {self.green_axis}"
            return

        if self.state == YELLOW:
            self.state = RED
            self.phase_remaining = self.ALL_RED_SECONDS
            self.last_decision = "Rojo de despeje"
            return

        # Al terminar el despeje, este agente elige la siguiente fase.
        if self.model.adaptive:
            next_axis = self._choose_next_axis()
        else:
            next_axis = NS if self.last_axis == EW else EW
        self.green_axis = next_axis
        self.last_axis = next_axis
        self.state = GREEN

        extra = self.pending_extension[next_axis]
        self.pending_extension[next_axis] = 0.0
        self.phase_remaining = min(
            self.MAX_GREEN_SECONDS,
            self.GREEN_SECONDS + extra,
        )
        self.last_decision = (
            f"Verde {next_axis}"
            + (f" (+{extra:.1f}s)" if extra > 0 else "")
        )


# ---------------------------------------------------------------------------
# Agente vehículo
# ---------------------------------------------------------------------------

class VehicleAgent(mesa.Agent):
    """Vehículo con origen, destino, velocidad y conducción segura."""

    # El centro del auto queda antes de la cebra; su frente llega a la
    # línea de detención sin invadir el paso peatonal.
    STOP_LINE_OFFSET = 67.0
    COMMIT_DISTANCE = 29.0
    CLEARANCE_DISTANCE = 22.0
    MIN_CAR_GAP = 24.0

    def __init__(
        self,
        model: "TrafficModel",
        origin: tuple[float, float],
        destination: tuple[float, float],
        axis: str,
        lane_index: int,
        heading: int,
        speed: float,
        color: tuple[int, int, int],
    ):
        super().__init__(model)

        self.origin = origin
        self.destination = destination
        self.x, self.y = origin
        self.axis = axis
        self.lane_index = lane_index
        self.heading = heading
        self.speed = speed
        self.color = color

        self.waiting_time = 0.0
        self.spawn_time = self.model.simulation_time
        self.is_stopped = False
        self.committed_signal_id: Optional[int] = None
        self.last_turn_signal_id: Optional[int] = None
        self.turns_remaining = self.model.route_random.randint(1, 2)

        self.model.vehicles.append(self)

    def _axis_position(self) -> float:
        return self.x if self.axis == EW else self.y

    def _signal_axis_position(self, signal: TrafficLightAgent) -> float:
        return signal.x if self.axis == EW else signal.y

    def _nearest_signal_ahead(
        self,
    ) -> tuple[Optional[TrafficLightAgent], float]:
        nearest = None
        nearest_distance = float("inf")
        position = self._axis_position()

        for signal in self.model.signals:
            if self.axis == EW and signal.row != self.lane_index:
                continue
            if self.axis == NS and signal.col != self.lane_index:
                continue

            distance = (
                self._signal_axis_position(signal) - position
            ) * self.heading

            if 0 <= distance < nearest_distance:
                nearest = signal
                nearest_distance = distance

        return nearest, nearest_distance

    def _update_intersection_commitment(self) -> None:
        """
        Un vehículo que ya alcanzó el punto de decisión con verde termina de
        despejar la intersección aunque el semáforo cambie a amarillo.
        """
        if self.committed_signal_id is not None:
            committed = self.model.signals_by_id.get(
                self.committed_signal_id
            )
            if committed is None:
                self.committed_signal_id = None
                self.last_turn_signal_id = None
            else:
                passed_distance = (
                    self._axis_position()
                    - self._signal_axis_position(committed)
                ) * self.heading
                if passed_distance > self.CLEARANCE_DISTANCE:
                    self.committed_signal_id = None
                    self.last_turn_signal_id = None

        signal, distance = self._nearest_signal_ahead()
        if (
            self.committed_signal_id is None
            and signal is not None
            and distance <= self.COMMIT_DISTANCE
            and signal.can_pass(self.axis)
        ):
            self.committed_signal_id = signal.unique_id

    def _distance_to_vehicle_ahead(self) -> float:
        nearest_gap = float("inf")
        position = self._axis_position()

        for other in self.model.vehicles:
            if other is self:
                continue
            if (
                other.axis != self.axis
                or other.lane_index != self.lane_index
                or other.heading != self.heading
            ):
                continue

            gap = (other._axis_position() - position) * self.heading
            if 0 < gap < nearest_gap:
                nearest_gap = gap

        return nearest_gap

    def _turn_right_at(self, signal: TrafficLightAgent) -> None:
        if self.axis == EW:
            self.axis = NS
            self.lane_index = signal.col
            self.x = signal.x + (-8 if self.heading > 0 else 8)
            self.y = signal.y
        else:
            self.axis = EW
            self.heading = -self.heading
            self.lane_index = signal.row
            self.x = signal.x
            self.y = signal.y + (-8 if self.heading > 0 else 8)

        if self.axis == EW:
            target_x = CITY_WIDTH - 18.0 if self.heading > 0 else 18.0
            self.destination = (target_x, self.y)
        else:
            target_y = CITY_HEIGHT - 18.0 if self.heading > 0 else 18.0
            self.destination = (self.x, target_y)

        self.turns_remaining -= 1
        self.last_turn_signal_id = signal.unique_id
        self.model.turns_taken += 1

    def _move_along_current_axis(self, distance: float) -> None:
        if self.axis == EW:
            self.x += self.heading * distance
        else:
            self.y += self.heading * distance

    def step_vehicle(self, dt: float) -> None:
        self._update_intersection_commitment()

        movement = self.speed * dt

        # Mantiene una distancia mínima con el vehículo que va adelante.
        gap = self._distance_to_vehicle_ahead()
        if gap != float("inf"):
            movement = min(
                movement,
                max(0.0, gap - self.MIN_CAR_GAP),
            )

        # Si el siguiente semáforo no permite el paso, frena antes de la línea.
        signal, distance = self._nearest_signal_ahead()
        if (
            signal is not None
            and signal.unique_id != self.committed_signal_id
            and not signal.can_pass(self.axis)
        ):
            distance_to_stop_line = max(
                0.0,
                distance - self.STOP_LINE_OFFSET,
            )
            movement = min(movement, distance_to_stop_line)

        if (
            signal is not None
            and signal.unique_id == self.committed_signal_id
            and signal.unique_id != self.last_turn_signal_id
            and movement > 0.1
            and distance <= movement
            and self.turns_remaining > 0
            and self.model.route_random.random()
            < self.model.TURN_PROBABILITY
        ):
            distance_after_intersection = max(0.0, movement - distance)
            if self.axis == EW:
                self.x = signal.x
            else:
                self.y = signal.y

            self._turn_right_at(signal)
            gap_after_turn = self._distance_to_vehicle_ahead()
            if gap_after_turn != float("inf"):
                distance_after_intersection = min(
                    distance_after_intersection,
                    max(0.0, gap_after_turn - self.MIN_CAR_GAP),
                )
            self._move_along_current_axis(distance_after_intersection)
            self.is_stopped = (
                distance + distance_after_intersection < 0.1
            )
            if self.is_stopped:
                self.waiting_time += dt
            return

        old_position = self._axis_position()
        target_position = (
            self.destination[0] if self.axis == EW else self.destination[1]
        )
        remaining = (target_position - old_position) * self.heading

        if remaining <= movement:
            self.x, self.y = self.destination
            self.waiting_time += 0.0
            self.model.completed_wait_sum += self.waiting_time
            self.model.completed_trip_time_sum += (
                self.model.simulation_time - self.spawn_time
            )
            self.model.completed_count += 1
            self.remove()
            return

        movement = max(0.0, movement)
        self._move_along_current_axis(movement)

        self.is_stopped = movement < 0.1
        if self.is_stopped:
            self.waiting_time += dt

    def remove(self) -> None:
        if self in self.model.vehicles:
            self.model.vehicles.remove(self)
        super().remove()


# ---------------------------------------------------------------------------
# Modelo Mesa: crea agentes y enruta mensajes entre vecinos
# ---------------------------------------------------------------------------

class TrafficModel(mesa.Model):
    """Modelo de tráfico con una cuadrícula de intersecciones."""

    BASE_SPAWN_RATE = 1.25
    HEAVY_SPAWN_RATE = 5.5
    HEAVY_DURATION = 12.0
    INITIAL_VEHICLE_COUNT = 8
    METRICS_INTERVAL = 0.5
    ALERT_DISPLAY_SECONDS = 1.6
    TURN_PROBABILITY = 0.28

    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        seed: Optional[int] = 7,
        adaptive: bool = True,
    ):
        if rows < 2 or cols < 2:
            raise ValueError("La cuadrícula debe tener al menos 2x2 intersecciones.")

        super().__init__(seed=seed)

        self.seed_value = seed
        self.adaptive = adaptive
        self.arrival_random = random.Random(seed)
        self.route_random = random.Random(None if seed is None else seed + 1)
        self.rows = rows
        self.cols = cols
        self.road_x = [
            ROAD_MARGIN_X
            + col * (CITY_WIDTH * 0.61) / (cols - 1)
            for col in range(cols)
        ]
        self.road_y = [
            ROAD_MARGIN_Y
            + row * (CITY_HEIGHT * 0.544) / (rows - 1)
            for row in range(rows)
        ]

        self.simulation_time = 0.0
        self.heavy_until = 0.0
        self.alerts_sent = 0
        self.completed_count = 0
        self.completed_wait_sum = 0.0
        self.completed_trip_time_sum = 0.0
        self.turns_taken = 0
        self.max_queue_length = 0
        self.metrics_elapsed = 0.0
        self.metrics_history: list[dict] = []
        self.active_alerts: list[dict] = []
        self.last_event = "Esperando tráfico"

        # Estas listas son referencias de conveniencia; Mesa sigue registrando
        # todos los agentes en model.agents.
        self.signals: list[TrafficLightAgent] = []
        self.vehicles: list[VehicleAgent] = []
        self.signals_by_position: dict[
            tuple[int, int], TrafficLightAgent
        ] = {}
        self.signals_by_id: dict[int, TrafficLightAgent] = {}

        for row in range(rows):
            for col in range(cols):
                signal = TrafficLightAgent(self, row, col)
                self.signals.append(signal)
                self.signals_by_position[(row, col)] = signal
                signal_id = signal.unique_id
                if signal_id is None:
                    raise RuntimeError(
                        "Mesa no asignó un identificador al semáforo."
                    )
                self.signals_by_id[signal_id] = signal

        # Algunos autos iniciales hacen visible la simulación desde el inicio.
        for _ in range(self.INITIAL_VEHICLE_COUNT):
            self._spawn_vehicle()

        for signal in self.signals:
            signal.sense_local_traffic()
        self._record_metrics()

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicles)

    @property
    def average_wait_time(self) -> float:
        active_wait = sum(vehicle.waiting_time for vehicle in self.vehicles)
        total_vehicles = self.completed_count + len(self.vehicles)
        if total_vehicles == 0:
            return 0.0
        return (
            self.completed_wait_sum + active_wait
        ) / total_vehicles

    @property
    def average_trip_time(self) -> float:
        if self.completed_count == 0:
            return 0.0
        return self.completed_trip_time_sum / self.completed_count

    @property
    def total_queue_length(self) -> int:
        return sum(signal.urgency for signal in self.signals)

    def _metrics_snapshot(self) -> dict:
        return {
            "seed": self.seed_value,
            "control_mode": "adaptativo" if self.adaptive else "tiempo_fijo",
            "time_s": round(self.simulation_time, 2),
            "queue_length": self.total_queue_length,
            "max_queue_length": self.max_queue_length,
            "average_wait_s": round(self.average_wait_time, 3),
            "average_trip_s": round(self.average_trip_time, 3),
            "vehicles_active": self.vehicle_count,
            "trips_completed": self.completed_count,
            "turns_taken": self.turns_taken,
            "alerts_sent": self.alerts_sent,
            "heavy_traffic": self.heavy_traffic_active,
        }

    def _record_metrics(self) -> None:
        self.max_queue_length = max(
            self.max_queue_length,
            self.total_queue_length,
        )
        self.metrics_history.append(self._metrics_snapshot())

    def register_flow_alert(
        self,
        sender: TrafficLightAgent,
        receiver: TrafficLightAgent,
    ) -> None:
        self.active_alerts.append({
            "sender_id": sender.unique_id,
            "receiver_id": receiver.unique_id,
            "sent_at": self.simulation_time,
            "expires_at": self.simulation_time + self.ALERT_DISPLAY_SECONDS,
        })

    def export_metrics_csv(self) -> str:
        if (
            not self.metrics_history
            or self.metrics_history[-1]["time_s"]
            != round(self.simulation_time, 2)
        ):
            self._record_metrics()

        filename = (
            f"metricas_trafico_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )
        path = Path.cwd() / filename
        with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(self.metrics_history[-1].keys()),
            )
            writer.writeheader()
            writer.writerows(self.metrics_history)

        self.last_event = f"CSV exportado: {filename}"
        return filename

    @property
    def heavy_traffic_active(self) -> bool:
        return self.simulation_time < self.heavy_until

    @property
    def spawn_rate(self) -> float:
        return (
            self.HEAVY_SPAWN_RATE
            if self.heavy_traffic_active
            else self.BASE_SPAWN_RATE
        )

    def count_approaching(
        self,
        signal: TrafficLightAgent,
        axis: str,
        heading: Optional[int] = None,
    ) -> int:
        """Estimación local de los vehículos que se aproximan al semáforo."""
        count = 0

        for vehicle in self.vehicles:
            if vehicle.committed_signal_id == signal.unique_id:
                continue
            if vehicle.axis != axis:
                continue
            if heading is not None and vehicle.heading != heading:
                continue
            if axis == EW and vehicle.lane_index != signal.row:
                continue
            if axis == NS and vehicle.lane_index != signal.col:
                continue

            vehicle_position = (
                vehicle.x if axis == EW else vehicle.y
            )
            signal_position = signal.x if axis == EW else signal.y
            distance = (signal_position - vehicle_position) * vehicle.heading

            if 0 <= distance <= signal.QUEUE_RANGE:
                count += 1

        return count

    def inject_heavy_traffic(self) -> None:
        """Activa una llegada de tráfico pesado durante varios segundos."""
        self.heavy_until = max(
            self.heavy_until,
            self.simulation_time + self.HEAVY_DURATION,
        )
        self.last_event = "Inyección de tráfico pesado activada"

    def _can_spawn_at(
        self,
        axis: str,
        lane_index: int,
        heading: int,
        position: float,
    ) -> bool:
        for vehicle in self.vehicles:
            if (
                vehicle.axis == axis
                and vehicle.lane_index == lane_index
                and vehicle.heading == heading
                and abs(vehicle._axis_position() - position) < 30.0
            ):
                return False
        return True

    def _spawn_vehicle(self) -> None:
        axis = self.arrival_random.choice([EW, NS])
        heading = self.arrival_random.choice([-1, 1])

        if axis == EW:
            row = self.arrival_random.randrange(self.rows)
            lane_index = row
            y = self.road_y[row] + (-8 if heading > 0 else 8)
            x = 18.0 if heading > 0 else CITY_WIDTH - 18.0
            origin = (x, y)
            destination = (
                (CITY_WIDTH - 18.0, y)
                if heading > 0
                else (18.0, y)
            )
            spawn_position = x
        else:
            col = self.arrival_random.randrange(self.cols)
            lane_index = col
            x = self.road_x[col] + (-8 if heading > 0 else 8)
            y = 18.0 if heading > 0 else CITY_HEIGHT - 18.0
            origin = (x, y)
            destination = (
                (x, CITY_HEIGHT - 18.0)
                if heading > 0
                else (x, 18.0)
            )
            spawn_position = y

        speed = self.arrival_random.uniform(82.0, 112.0)
        color = self.arrival_random.choice(VEHICLE_COLORS)
        if not self._can_spawn_at(
            axis, lane_index, heading, spawn_position
        ):
            return

        VehicleAgent(
            model=self,
            origin=origin,
            destination=destination,
            axis=axis,
            lane_index=lane_index,
            heading=heading,
            speed=speed,
            color=color,
        )

    def _advance_once(self, dt: float) -> None:
        self.simulation_time += dt

        # Llegada aleatoria. En tráfico pesado aumenta la tasa de generación.
        if self.arrival_random.random() < self.spawn_rate * dt:
            self._spawn_vehicle()

        # Cada semáforo mide su propia cola, luego decide si debe alertar
        # a sus vecinos, y finalmente procesa los mensajes recibidos.
        for signal in self.signals:
            signal.sense_local_traffic()

        self.active_alerts = [
            alert
            for alert in self.active_alerts
            if alert["expires_at"] > self.simulation_time
        ]
        for signal in self.signals:
            signal.maybe_broadcast_flow_alert()
        for signal in self.signals:
            signal.update_phase(dt)

        # El recorrido de una copia permite que un auto salga del modelo
        # durante este mismo paso sin alterar la iteración.
        for vehicle in tuple(self.vehicles):
            vehicle.step_vehicle(dt)

        # Refresca la vista de las métricas después de mover los vehículos.
        for signal in self.signals:
            signal.sense_local_traffic()
        self.max_queue_length = max(
            self.max_queue_length,
            self.total_queue_length,
        )
        self.metrics_elapsed += dt
        if self.metrics_elapsed >= self.METRICS_INTERVAL:
            self.metrics_elapsed %= self.METRICS_INTERVAL
            self._record_metrics()

    def advance(self, dt: float) -> None:
        """Avanza la simulación con subpasos para mantener movimientos seguros."""
        if dt <= 0:
            return

        substeps = max(1, math.ceil(dt / 0.05))
        sub_dt = dt / substeps
        for _ in range(substeps):
            self._advance_once(sub_dt)

    def step(self) -> None:
        """Método estándar de Mesa: avanza un segundo de tiempo simulado."""
        self.advance(1.0)


# ---------------------------------------------------------------------------
# Ventana pygame
# ---------------------------------------------------------------------------

class SimulationWindow:
    """Interfaz en vivo con mapa, métricas y botón de tráfico pesado."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(
            "Centro de Control de Tráfico — Red urbana 3x3"
        )
        display_info = pygame.display.Info()
        desktop_width = display_info.current_w or DISPLAY_SIZE[0]
        desktop_height = display_info.current_h or DISPLAY_SIZE[1]
        initial_scale = min(
            1.0,
            max(1, desktop_width - 40) / DISPLAY_SIZE[0],
            max(1, desktop_height - 80) / DISPLAY_SIZE[1],
        )
        initial_size = (
            max(640, int(DISPLAY_SIZE[0] * initial_scale)),
            max(480, int(DISPLAY_SIZE[1] * initial_scale)),
        )
        self.display = pygame.display.set_mode(initial_size, pygame.RESIZABLE)
        self.screen = pygame.Surface(WINDOW_SIZE)
        self.map_surface = pygame.Surface((CITY_WIDTH, CITY_HEIGHT))
        self.panel_surface = pygame.Surface((PANEL_WIDTH, PANEL_HEIGHT))
        self.header_surface = pygame.Surface((DISPLAY_SIZE[0], DISPLAY_HEADER_HEIGHT))
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("segoeui", 23, bold=True)
        self.font = pygame.font.SysFont("segoeui", 17)
        self.small_font = pygame.font.SysFont("segoeui", 11)
        self.tiny_font = pygame.font.SysFont("segoeui", 9)
        self.chart_font = pygame.font.SysFont("segoeui", 10)
        self.tooltip_font = pygame.font.SysFont("segoeui", 9)
        self.section_font = pygame.font.SysFont("segoeui", 9, bold=True)
        self.metric_font = pygame.font.SysFont("segoeui", 16, bold=True)
        self.button_font = pygame.font.SysFont("segoeui", 9, bold=True)
        self.signal_font = pygame.font.SysFont("segoeui", 8, bold=True)
        self.header_title_font = pygame.font.SysFont("segoeui", 17, bold=True)
        self.header_sub_font = pygame.font.SysFont("segoeui", 10, bold=True)
        self.header_meta_font = pygame.font.SysFont("segoeui", 10)
        self.signal_badges: list[tuple[pygame.Rect, TrafficLightAgent]] = []
        self.signal_head_rects: list[tuple[pygame.Rect, TrafficLightAgent]] = []

        self.seed = 7
        self.model = TrafficModel(rows=3, cols=3, seed=self.seed, adaptive=True)
        self.control_model = TrafficModel(
            rows=3,
            cols=3,
            seed=self.seed,
            adaptive=False,
        )
        self.running = True
        self.paused = False
        self.speed_levels = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
        self.speed_index = 3

        panel_x = 18
        button_width = PANEL_WIDTH - 36
        self.heavy_button = pygame.Rect(
            panel_x,
            490,
            button_width,
            36,
        )
        self.pause_button = pygame.Rect(panel_x, 535, 156, 34)
        self.reset_button = pygame.Rect(panel_x + 168, 535, 156, 34)
        self.speed_down_button = pygame.Rect(panel_x, 580, 32, 28)
        self.speed_up_button = pygame.Rect(panel_x + 292, 580, 32, 28)
        self.export_button = pygame.Rect(panel_x, 610, 128, 20)

    @property
    def speed_multiplier(self) -> float:
        return self.speed_levels[self.speed_index]

    def _fitted_output_rect(self) -> pygame.Rect:
        """Área donde cabe el diseño 16:10 sin deformarlo."""
        window_width, window_height = self.display.get_size()
        scale = min(
            window_width / DISPLAY_SIZE[0],
            window_height / DISPLAY_SIZE[1],
        )
        width = max(1, round(DISPLAY_SIZE[0] * scale))
        height = max(1, round(DISPLAY_SIZE[1] * scale))
        return pygame.Rect(
            (window_width - width) // 2,
            (window_height - height) // 2,
            width,
            height,
        )

    def _window_to_design(self, position: tuple[int, int]) -> Optional[tuple[int, int]]:
        output = self._fitted_output_rect()
        if not output.collidepoint(position):
            return None
        return (
            int((position[0] - output.x) * DISPLAY_SIZE[0] / output.width),
            int((position[1] - output.y) * DISPLAY_SIZE[1] / output.height),
        )

    def _set_speed(self, direction: int) -> None:
        self.speed_index = max(
            0,
            min(len(self.speed_levels) - 1, self.speed_index + direction),
        )

    def _draw_button(
        self,
        rect: pygame.Rect,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(self.screen, color, rect, border_radius=7)
        pygame.draw.rect(
            self.screen,
            (206, 218, 224),
            rect,
            width=1,
            border_radius=7,
        )
        label_surface = self.button_font.render(label, True, (255, 255, 255))
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))

    def _reset_model(self) -> None:
        self.model = TrafficModel(rows=3, cols=3, seed=self.seed, adaptive=True)
        self.control_model = TrafficModel(
            rows=3,
            cols=3,
            seed=self.seed,
            adaptive=False,
        )

    def _export_metrics_csv(self) -> str:
        for scenario in (self.model, self.control_model):
            if (
                not scenario.metrics_history
                or scenario.metrics_history[-1]["time_s"]
                != round(scenario.simulation_time, 2)
            ):
                scenario._record_metrics()

        filename = (
            f"comparacion_trafico_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )
        path = Path.cwd() / filename
        with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            fieldnames = list(self.model.metrics_history[-1].keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for scenario in (self.model, self.control_model):
                writer.writerows(scenario.metrics_history)

        self.model.last_event = f"CSV comparativo: {filename}"
        return filename

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int] = TEXT_COLOR,
    ) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def _draw_city_blocks(self) -> None:
        """Añade manzanas claras y edificios alrededor de las calzadas."""
        half_road = ROAD_WIDTH // 2
        x_segments = [(0, int(self.model.road_x[0] - half_road))]
        x_segments.extend(
            (int(left + half_road), int(right - half_road))
            for left, right in zip(self.model.road_x, self.model.road_x[1:])
        )
        x_segments.append((int(self.model.road_x[-1] + half_road), CITY_WIDTH))

        y_segments = [(0, int(self.model.road_y[0] - half_road))]
        y_segments.extend(
            (int(top + half_road), int(bottom - half_road))
            for top, bottom in zip(self.model.road_y, self.model.road_y[1:])
        )
        y_segments.append((int(self.model.road_y[-1] + half_road), CITY_HEIGHT))

        for row_index, (top, bottom) in enumerate(y_segments):
            for col_index, (left, right) in enumerate(x_segments):
                width = right - left
                height = bottom - top
                if width >= 125 and height >= 105:
                    buildings = (
                        pygame.Rect(
                            left + 12,
                            top + 14,
                            int(width * 0.35),
                            int(height * 0.48),
                        ),
                        pygame.Rect(
                            left + int(width * 0.55),
                            top + int(height * 0.18),
                            int(width * 0.36),
                            int(height * 0.53),
                        ),
                    )
                    for building in buildings:
                        shadow = building.move(3, 4)
                        pygame.draw.rect(
                            self.screen,
                            (185, 198, 201),
                            shadow,
                            border_radius=5,
                        )
                        pygame.draw.rect(
                            self.screen,
                            (240, 244, 243),
                            building,
                            border_radius=5,
                        )
                        pygame.draw.rect(
                            self.screen,
                            (184, 198, 201),
                            building,
                            width=1,
                            border_radius=5,
                        )
                        pygame.draw.line(
                            self.screen,
                            (213, 224, 225),
                            (building.x + 7, building.y + 8),
                            (building.right - 7, building.y + 8),
                            3,
                        )
                elif width > 55 and height > 45:
                    tree_x = left + width // 2
                    tree_y = top + height // 2
                    for dx, dy in ((-10, -7), (10, 8)):
                        pygame.draw.circle(
                            self.screen,
                            (173, 201, 183),
                            (tree_x + dx, tree_y + dy),
                            6,
                        )
                        pygame.draw.circle(
                            self.screen,
                            (147, 181, 160),
                            (tree_x + dx, tree_y + dy),
                            6,
                            1,
                        )

    def _draw_map(self) -> None:
        display = self.screen
        self.screen = self.map_surface
        self.screen.fill(GRASS, (0, 0, CITY_WIDTH, CITY_HEIGHT))
        self._draw_city_blocks()

        # Calzadas horizontales y sus marcas viales discontinuas.
        for y in self.model.road_y:
            road_y = int(y)
            pygame.draw.rect(
                self.screen,
                ASPHALT,
                (0, road_y - ROAD_WIDTH // 2, CITY_WIDTH, ROAD_WIDTH),
            )
            for x in range(8, CITY_WIDTH, 42):
                pygame.draw.line(
                    self.screen,
                    ROAD_MARKING,
                    (x, road_y),
                    (x + 20, road_y),
                    2,
                )

        # Calzadas verticales y marcas viales.
        for x in self.model.road_x:
            road_x = int(x)
            pygame.draw.rect(
                self.screen,
                ASPHALT,
                (road_x - ROAD_WIDTH // 2, 0, ROAD_WIDTH, CITY_HEIGHT),
            )
            for y in range(8, CITY_HEIGHT, 42):
                pygame.draw.line(
                    self.screen,
                    ROAD_MARKING,
                    (road_x, y),
                    (road_x, y + 20),
                    2,
                )

        self._draw_queue_indicators()
        self._draw_intersection_markings()

        # Vehículos: rectángulos orientados según el eje de circulación.
        for vehicle in self.model.vehicles:
            if vehicle.axis == EW:
                rect = pygame.Rect(
                    int(vehicle.x - 9),
                    int(vehicle.y - 5),
                    18,
                    10,
                )
            else:
                rect = pygame.Rect(
                    int(vehicle.x - 5),
                    int(vehicle.y - 9),
                    10,
                    18,
                )
            pygame.draw.rect(
                self.screen,
                vehicle.color,
                rect,
                border_radius=3,
            )
            pygame.draw.rect(
                self.screen,
                (25, 29, 34),
                rect,
                width=1,
                border_radius=3,
            )
            if vehicle.axis == EW:
                window = pygame.Rect(rect.x + 4, rect.y + 2, 9, 6)
            else:
                window = pygame.Rect(rect.x + 2, rect.y + 4, 6, 9)
            pygame.draw.rect(self.screen, (49, 77, 91), window, border_radius=2)
            pygame.draw.rect(self.screen, (158, 191, 198), window, width=1, border_radius=2)

        self._draw_alert_links()
        self._draw_signals()
        self._draw_map_scale()
        self._draw_map_legend()
        self.screen = display
        display.blit(self.map_surface, (0, HEADER_HEIGHT))

    def _draw_alert_links(self) -> None:
        for alert in self.model.active_alerts:
            sender = self.model.signals_by_id.get(alert["sender_id"])
            receiver = self.model.signals_by_id.get(alert["receiver_id"])
            if sender is None or receiver is None:
                continue

            duration = max(
                0.01,
                alert["expires_at"] - alert["sent_at"],
            )
            progress = min(
                1.0,
                max(0.0, (self.model.simulation_time - alert["sent_at"]) / duration),
            )
            start = (int(sender.x), int(sender.y))
            end = (int(receiver.x), int(receiver.y))
            pygame.draw.line(self.screen, (28, 173, 211), start, end, 3)

            pulse = (
                int(start[0] + (end[0] - start[0]) * progress),
                int(start[1] + (end[1] - start[1]) * progress),
            )
            pygame.draw.circle(self.screen, (22, 30, 38), pulse, 8)
            pygame.draw.circle(self.screen, YELLOW_COLOR, pulse, 5)
            pygame.draw.circle(self.screen, (255, 255, 255), pulse, 2)

    def _draw_intersection_markings(self) -> None:
        """Dibuja pasos peatonales y líneas de detención en cada cruce."""
        crosswalk_color = (238, 238, 224)
        stop_color = (250, 246, 221)
        stop_marking_offset = VehicleAgent.STOP_LINE_OFFSET - 9

        for signal in self.model.signals:
            x = int(signal.x)
            y = int(signal.y)

            # Cruces peatonales en los cuatro accesos.
            for offset in (-42, 42):
                for stripe in (-14, -7, 0, 7, 14):
                    pygame.draw.line(
                        self.screen,
                        crosswalk_color,
                        (x + offset + stripe, y - ROAD_WIDTH // 2 + 4),
                        (x + offset + stripe, y + ROAD_WIDTH // 2 - 4),
                        3,
                    )
                    pygame.draw.line(
                        self.screen,
                        crosswalk_color,
                        (x - ROAD_WIDTH // 2 + 4, y + offset + stripe),
                        (x + ROAD_WIDTH // 2 - 4, y + offset + stripe),
                        3,
                    )

            # Líneas separadas por sentido para dejar visibles las dos vías.
            # Las marcas quedan justo antes de las cebras, fuera del cruce.
            for heading in (-1, 1):
                stop_x = round(x - heading * stop_marking_offset)
                stop_y = round(y - heading * stop_marking_offset)
                pygame.draw.line(
                    self.screen,
                    stop_color,
                    (stop_x, y - ROAD_WIDTH // 2 + 3),
                    (stop_x, y + ROAD_WIDTH // 2 - 3),
                    2,
                )
                pygame.draw.line(
                    self.screen,
                    stop_color,
                    (x - ROAD_WIDTH // 2 + 3, stop_y),
                    (x + ROAD_WIDTH // 2 - 3, stop_y),
                    2,
                )

    def _draw_queue_indicators(self) -> None:
        """Pinta una franja de longitud creciente en cada acceso congestionado."""
        for signal in self.model.signals:
            for axis in (EW, NS):
                for heading in (-1, 1):
                    count = self.model.count_approaching(signal, axis, heading)
                    if count <= 0:
                        continue

                    if count == 1:
                        color = (126, 207, 114)
                    elif count == 2:
                        color = (245, 196, 72)
                    elif count == 3:
                        color = (244, 139, 64)
                    else:
                        color = (237, 79, 76)

                    queue_length = min(
                        signal.QUEUE_RANGE,
                        VehicleAgent.STOP_LINE_OFFSET + min(count, 4) * 18,
                    )
                    if axis == EW:
                        lane_y = int(signal.y - 8 if heading > 0 else signal.y + 8)
                        stop_x = signal.x - heading * VehicleAgent.STOP_LINE_OFFSET
                        start_x = stop_x - heading * queue_length
                        pygame.draw.line(
                            self.screen,
                            color,
                            (int(start_x), lane_y),
                            (int(stop_x), lane_y),
                            5,
                        )
                    else:
                        lane_x = int(signal.x - 8 if heading > 0 else signal.x + 8)
                        stop_y = signal.y - heading * VehicleAgent.STOP_LINE_OFFSET
                        start_y = stop_y - heading * queue_length
                        pygame.draw.line(
                            self.screen,
                            color,
                            (lane_x, int(start_y)),
                            (lane_x, int(stop_y)),
                            5,
                        )

    def _draw_signals(self) -> None:
        color_by_state = {
            GREEN: GREEN_COLOR,
            YELLOW: YELLOW_COLOR,
            RED: RED_COLOR,
        }

        self.signal_badges = []
        self.signal_head_rects = []
        for signal in self.model.signals:
            x = int(signal.x)
            y = int(signal.y)
            head_layout = (
                (EW, -1, -1),
                (NS, 1, -1),
                (NS, -1, 1),
                (EW, 1, 1),
            )
            heads = []
            half_road = ROAD_WIDTH // 2
            for axis, side_x, side_y in head_layout:
                head_x = x + side_x * (half_road + 15) - 6
                head_y = y + side_y * (half_road + 15) - 15
                head = pygame.Rect(head_x, head_y, 12, 30)
                pygame.draw.rect(
                    self.screen,
                    (25, 34, 43),
                    head,
                    border_radius=4,
                )
                pygame.draw.rect(
                    self.screen,
                    (117, 133, 142),
                    head,
                    width=1,
                    border_radius=4,
                )
                active_state = signal.state_for(axis)
                for index, state in enumerate((RED, YELLOW, GREEN)):
                    light_color = (
                        color_by_state[state]
                        if active_state == state
                        else (62, 72, 80)
                    )
                    pygame.draw.circle(
                        self.screen,
                        light_color,
                        (head.centerx, head.y + 5 + index * 10),
                        3,
                    )
                self.signal_head_rects.append((head, signal))
                heads.append((head, axis))

            seconds_left = max(0, math.ceil(signal.phase_remaining))
            phase_color = color_by_state[signal.state]
            state_label = {GREEN: "VERDE", YELLOW: "AMAR.", RED: "ROJO"}[signal.state]
            if signal.state == RED:
                phase_text = f"ROJO · {seconds_left}s"
            else:
                axis_label = "E-O" if signal.green_axis == EW else "N-S"
                phase_text = f"{axis_label} {state_label} · {seconds_left}s"

            badge_width = 66 if signal.col == 0 else 78
            badge_height = 42
            if signal.col == 0:
                badge_x = 4
            elif signal.col == 1:
                badge_x = 134
            else:
                badge_x = 586
            if signal.row == 0:
                badge_y = int(y + 65)
            else:
                badge_y = int(y - 100)
            badge = pygame.Rect(badge_x, badge_y, badge_width, badge_height)

            nearest_head, _ = min(
                heads,
                key=lambda item: math.dist(badge.center, item[0].center),
            )
            pygame.draw.line(
                self.screen,
                (134, 158, 151),
                badge.center,
                nearest_head.center,
                1,
            )
            pygame.draw.rect(
                self.screen,
                (25, 31, 39),
                badge,
                border_radius=6,
            )
            pygame.draw.rect(
                self.screen,
                phase_color,
                badge,
                width=1,
                border_radius=6,
            )
            self._draw_text(
                phase_text,
                badge_x + 4,
                badge_y + 5,
                self.signal_font,
                phase_color,
            )
            self._draw_text(
                f"E:{signal.queue_counts[EW]} N:{signal.queue_counts[NS]}",
                badge_x + 4,
                badge_y + 23,
                self.tiny_font,
                MUTED_TEXT,
            )
            self.signal_badges.append((badge, signal))

        self._draw_signal_tooltip()

    def _draw_signal_tooltip(self) -> None:
        design_mouse = self._window_to_design(pygame.mouse.get_pos())
        if design_mouse is None:
            return
        mouse_x, mouse_y = design_mouse
        if not (
            0 <= mouse_x < DISPLAY_MAP_SIZE[0]
            and DISPLAY_HEADER_HEIGHT <= mouse_y < DISPLAY_SIZE[1]
        ):
            return
        mouse = (
            int(mouse_x * CITY_WIDTH / DISPLAY_MAP_SIZE[0]),
            int((mouse_y - DISPLAY_HEADER_HEIGHT) * CITY_HEIGHT / DISPLAY_MAP_SIZE[1]),
        )
        hovered_signal = None
        for badge, signal in self.signal_badges:
            if badge.collidepoint(mouse):
                hovered_signal = signal
                break
        if hovered_signal is None:
            for head, signal in self.signal_head_rects:
                if head.collidepoint(mouse):
                    hovered_signal = signal
                    break
        if hovered_signal is None:
            return

        signal = hovered_signal
        axis_label = "E-O" if signal.green_axis == EW else "N-S"
        state_label = {
            GREEN: "verde",
            YELLOW: "amarillo",
            RED: "rojo",
        }[signal.state]
        lines = (
            f"Cruce {signal.row + 1}-{signal.col + 1} · {axis_label}: {state_label}",
            f"Cuenta regresiva: {max(0, math.ceil(signal.phase_remaining))} s",
            f"Cola E-O: {signal.queue_counts[EW]}   N-S: {signal.queue_counts[NS]}",
            f"Urgencia: {signal.urgency}   Alertas: {self.model.alerts_sent}",
            f"Extensiones E-O/N-S: {signal.pending_extension[EW]:.1f}/{signal.pending_extension[NS]:.1f} s",
            f"Decisión: {signal.last_decision[:28]}",
        )
        tooltip_width = 260
        tooltip_height = 116
        tooltip_x = min(max(8, mouse[0] + 14), CITY_WIDTH - tooltip_width - 8)
        tooltip_y = min(max(8, mouse[1] + 14), CITY_HEIGHT - tooltip_height - 8)
        tooltip = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)
        pygame.draw.rect(self.screen, (18, 24, 31), tooltip, border_radius=7)
        pygame.draw.rect(self.screen, (123, 153, 169), tooltip, width=1, border_radius=7)
        for index, line in enumerate(lines):
            self._draw_text(
                line,
                tooltip.x + 9,
                tooltip.y + 7 + index * 17,
                self.tooltip_font,
                TEXT_COLOR if index == 0 else MUTED_TEXT,
            )

    def _draw_legend(self) -> None:
        footer_y = CITY_HEIGHT
        pygame.draw.rect(
            self.screen,
            (29, 37, 46),
            (0, footer_y, WINDOW_SIZE[0], FOOTER_HEIGHT),
        )
        pygame.draw.line(
            self.screen,
            (75, 88, 99),
            (0, footer_y),
            (WINDOW_SIZE[0], footer_y),
            1,
        )

        y = footer_y + 23
        self._draw_text("Fase:", 16, footer_y + 14, self.chart_font, MUTED_TEXT)
        x = 62
        for label, color in (("verde", GREEN_COLOR), ("amarillo", YELLOW_COLOR), ("rojo", RED_COLOR)):
            pygame.draw.circle(self.screen, color, (x + 5, y), 5)
            self._draw_text(label, x + 14, footer_y + 14, self.chart_font)
            x += 75 if label != "amarillo" else 92

        # Sentidos indicados con flechas para distinguirlos del código de color.
        pygame.draw.line(self.screen, (210, 218, 225), (330, y), (372, y), 2)
        pygame.draw.polygon(self.screen, (210, 218, 225), [(330, y), (337, y - 4), (337, y + 4)])
        pygame.draw.polygon(self.screen, (210, 218, 225), [(372, y), (365, y - 4), (365, y + 4)])
        self._draw_text("E-O", 380, footer_y + 14, self.chart_font)
        pygame.draw.line(self.screen, (210, 218, 225), (443, y + 7), (443, y - 7), 2)
        pygame.draw.polygon(self.screen, (210, 218, 225), [(443, y + 7), (439, y + 1), (447, y + 1)])
        pygame.draw.polygon(self.screen, (210, 218, 225), [(443, y - 7), (439, y), (447, y)])
        self._draw_text("N-S", 453, footer_y + 14, self.chart_font)

        pygame.draw.line(self.screen, (28, 173, 211), (515, y), (555, y), 3)
        pygame.draw.circle(self.screen, YELLOW_COLOR, (535, y), 4)
        self._draw_text("mensajes entre semáforos", 563, footer_y + 14, self.chart_font)

        self._draw_text("Fila:", 758, footer_y + 14, self.chart_font, MUTED_TEXT)
        queue_colors = (
            (126, 207, 114),
            (245, 196, 72),
            (244, 139, 64),
            (237, 79, 76),
        )
        for index, color in enumerate(queue_colors):
            pygame.draw.line(
                self.screen,
                color,
                (801 + index * 17, y),
                (813 + index * 17, y),
                5,
            )
        self._draw_text("corta → larga", 876, footer_y + 14, self.chart_font)

    def _draw_map_legend(self) -> None:
        """Leyenda compacta sobre el borde inferior del mapa."""
        legend = pygame.Rect(
            16,
            CITY_HEIGHT - 47,
            450,
            38,
        )
        pygame.draw.rect(self.screen, (239, 243, 242), legend, border_radius=7)
        pygame.draw.rect(
            self.screen,
            (184, 198, 201),
            legend,
            width=1,
            border_radius=7,
        )
        color = (69, 84, 94)
        y = legend.centery

        self._draw_text("FASES", 26, y - 6, self.section_font, color)
        for x, label, phase_color in (
            (70, "verde", GREEN_COLOR),
            (127, "amarillo", YELLOW_COLOR),
            (196, "rojo", RED_COLOR),
        ):
            pygame.draw.circle(self.screen, phase_color, (x, y), 4)
            self._draw_text(label, x + 7, y - 6, self.tiny_font, color)

        pygame.draw.line(self.screen, (197, 207, 208), (252, y - 11), (252, y + 11), 1)
        self._draw_text("E-O", 260, y - 6, self.tiny_font, color)
        pygame.draw.line(self.screen, color, (286, y), (301, y), 2)
        pygame.draw.polygon(self.screen, color, [(286, y), (292, y - 4), (292, y + 4)])
        pygame.draw.polygon(self.screen, color, [(301, y), (295, y - 4), (295, y + 4)])
        self._draw_text("N-S", 305, y - 6, self.tiny_font, color)
        pygame.draw.line(self.screen, color, (329, y + 6), (329, y - 6), 2)
        pygame.draw.polygon(self.screen, color, [(329, y + 6), (325, y + 1), (333, y + 1)])
        pygame.draw.polygon(self.screen, color, [(329, y - 6), (325, y - 1), (333, y - 1)])

        pygame.draw.line(self.screen, (197, 207, 208), (340, y - 11), (340, y + 11), 1)
        self._draw_text("FILA", 348, y - 6, self.section_font, color)
        for index, queue_color in enumerate((GREEN_COLOR, YELLOW_COLOR, (244, 139, 64), RED_COLOR)):
            pygame.draw.line(self.screen, queue_color, (381 + index * 8, y), (387 + index * 8, y), 4)

        pygame.draw.line(self.screen, (197, 207, 208), (413, y - 11), (413, y + 11), 1)
        pygame.draw.line(self.screen, (28, 173, 211), (418, y), (430, y), 2)
        pygame.draw.polygon(self.screen, (28, 173, 211), [(430, y), (425, y - 4), (425, y + 4)])
        self._draw_text("MSG", 433, y - 6, self.tiny_font, color)

    def _draw_map_top_cards(self) -> None:
        left = pygame.Rect(16, 12, 164, 24)
        right = pygame.Rect(CITY_WIDTH - 180, 12, 164, 24)
        for card in (left, right):
            pygame.draw.rect(self.screen, (240, 244, 243), card, border_radius=6)
            pygame.draw.rect(self.screen, (184, 198, 201), card, width=1, border_radius=6)

        pygame.draw.circle(self.screen, GREEN_COLOR, (left.x + 12, left.y + 12), 4)
        self._draw_text("MAPA OPERATIVO", left.x + 23, left.y + 2, self.section_font, (48, 67, 77))
        self._draw_text("9 cruces · fases adaptativas", left.x + 23, left.y + 13, self.tiny_font, (86, 101, 110))

        self._draw_text("E-O ↔   N-S ↕", right.x + 10, right.y + 1, self.section_font, (48, 67, 77))
        self._draw_text("COLAS  ·  MENSAJES", right.x + 10, right.y + 13, self.tiny_font, (86, 101, 110))

        self._draw_text("ESCALA", CITY_WIDTH - 95, CITY_HEIGHT - 46, self.section_font, (112, 128, 132))
        scale_y = CITY_HEIGHT - 27
        pygame.draw.line(self.screen, (103, 120, 128), (CITY_WIDTH - 92, scale_y), (CITY_WIDTH - 48, scale_y), 2)
        pygame.draw.line(self.screen, (103, 120, 128), (CITY_WIDTH - 92, scale_y - 4), (CITY_WIDTH - 92, scale_y + 4), 2)
        pygame.draw.line(self.screen, (103, 120, 128), (CITY_WIDTH - 48, scale_y - 4), (CITY_WIDTH - 48, scale_y + 4), 2)
        self._draw_text("100 m", CITY_WIDTH - 44, scale_y - 6, self.tiny_font, (112, 128, 132))

    def _draw_map_scale(self) -> None:
        self._draw_text("ESCALA", CITY_WIDTH - 95, CITY_HEIGHT - 46, self.section_font, (112, 128, 132))
        scale_y = CITY_HEIGHT - 27
        pygame.draw.line(self.screen, (103, 120, 128), (CITY_WIDTH - 92, scale_y), (CITY_WIDTH - 48, scale_y), 2)
        pygame.draw.line(self.screen, (103, 120, 128), (CITY_WIDTH - 92, scale_y - 4), (CITY_WIDTH - 92, scale_y + 4), 2)
        pygame.draw.line(self.screen, (103, 120, 128), (CITY_WIDTH - 48, scale_y - 4), (CITY_WIDTH - 48, scale_y + 4), 2)
        self._draw_text("100 m", CITY_WIDTH - 44, scale_y - 6, self.tiny_font, (112, 128, 132))

    def _draw_header_info_cards(self) -> None:
        left = pygame.Rect(420, 13, 196, 38)
        right = pygame.Rect(628, 13, 196, 38)
        for card in (left, right):
            pygame.draw.rect(self.screen, (22, 38, 52), card, border_radius=7)
            pygame.draw.rect(self.screen, (48, 72, 88), card, width=1, border_radius=7)

        pygame.draw.circle(self.screen, GREEN_COLOR, (left.x + 13, left.centery), 4)
        self._draw_text("MAPA OPERATIVO", left.x + 25, left.y + 5, self.section_font, TEXT_COLOR)
        self._draw_text("9 cruces · fases adaptativas", left.x + 25, left.y + 20, self.tiny_font, MUTED_TEXT)

        self._draw_text("E-O ↔   N-S ↕", right.x + 11, right.y + 5, self.section_font, TEXT_COLOR)
        self._draw_text("COLAS · MENSAJES", right.x + 11, right.y + 20, self.tiny_font, MUTED_TEXT)

    def _draw_header(self) -> None:
        display = self.screen
        self.screen = self.header_surface
        pygame.draw.rect(
            self.screen,
            (15, 25, 37),
            (0, 0, DISPLAY_SIZE[0], DISPLAY_HEADER_HEIGHT),
        )
        pygame.draw.line(
            self.screen,
            (57, 74, 88),
            (0, DISPLAY_HEADER_HEIGHT - 1),
            (DISPLAY_SIZE[0], DISPLAY_HEADER_HEIGHT - 1),
            1,
        )

        logo = pygame.Rect(24, 15, 34, 34)
        pygame.draw.rect(self.screen, (19, 49, 64), logo, border_radius=8)
        pygame.draw.rect(self.screen, (44, 184, 212), logo, width=1, border_radius=8)
        nodes = ((33, 32), (41, 24), (49, 32), (41, 40))
        pygame.draw.lines(self.screen, (62, 207, 226), True, nodes, 2)
        for node in nodes:
            pygame.draw.circle(self.screen, (131, 230, 240), node, 2)

        self._draw_text("CENTRO DE CONTROL DE TRÁFICO", 72, 16, self.header_title_font, TEXT_COLOR)
        self._draw_text(
            "SIMULACIÓN DISTRIBUIDA  /  RED URBANA 3×3",
            72,
            39,
            self.header_sub_font,
            MUTED_TEXT,
        )
        self._draw_header_info_cards()

        live = pygame.Rect(1262, 17, 124, 30)
        pygame.draw.rect(self.screen, (17, 58, 49), live, border_radius=14)
        pygame.draw.rect(self.screen, (34, 144, 111), live, width=1, border_radius=14)
        pygame.draw.circle(self.screen, GREEN_COLOR, (1279, 32), 4)
        self._draw_text("RED EN VIVO", 1292, 25, self.header_sub_font, (185, 238, 219))
        self._draw_text(
            datetime.now().strftime("%d %b %Y · %H:%M:%S").upper(),
            1412,
            27,
            self.header_meta_font,
            MUTED_TEXT,
        )
        self.screen = display

    def _draw_queue_chart(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(
            self.screen,
            (18, 24, 31),
            rect,
            border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            (49, 66, 81),
            rect,
            width=1,
            border_radius=6,
        )
        datasets = (
            (self.model.metrics_history[-48:], (76, 190, 255)),
            (self.control_model.metrics_history[-48:], (255, 173, 82)),
        )
        if not any(samples for samples, _ in datasets):
            return

        left = rect.x + 28
        right = rect.right - 8
        top = rect.y + 18
        bottom = rect.bottom - 12
        max_value = max(
            4,
            max(
                row["queue_length"]
                for samples, _ in datasets
                for row in samples
            ),
        )

        for step in range(3):
            grid_y = top + (bottom - top) * step // 2
            pygame.draw.line(
                self.screen,
                (48, 58, 69),
                (left, grid_y),
                (right, grid_y),
                1,
            )

        self._draw_text(str(max_value), rect.x + 5, top - 6, self.chart_font, MUTED_TEXT)
        self._draw_text("0", rect.x + 13, bottom - 6, self.chart_font, MUTED_TEXT)

        for samples, color in datasets:
            if not samples:
                continue
            points = []
            denominator = max(1, len(samples) - 1)
            for index, row in enumerate(samples):
                px = left + (right - left) * index // denominator
                py = bottom - (bottom - top) * row["queue_length"] // max_value
                points.append((px, py))

            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
            pygame.draw.circle(self.screen, color, points[-1], 3)

        self._draw_text("A: adaptativo", rect.x + 72, rect.y + 2, self.chart_font, (154, 213, 245))
        self._draw_text("F: fijo", rect.right - 52, rect.y + 2, self.chart_font, (255, 205, 149))

    def _draw_panel(self) -> None:
        panel_x = CITY_WIDTH
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (panel_x, 0, PANEL_WIDTH, CITY_HEIGHT),
        )

        x = panel_x + 18
        panel_right = panel_x + PANEL_WIDTH - 18
        self._draw_text("TRÁFICO SMA", x, 14, self.title_font)
        self._draw_text(
            f"A/F · semilla {self.seed} · N cambia",
            x,
            45,
            self.small_font,
            MUTED_TEXT,
        )
        pygame.draw.line(
            self.screen,
            (66, 75, 86),
            (x, 68),
            (panel_right, 68),
            1,
        )

        self._draw_text("MÉTRICAS EN VIVO", x, 78, self.small_font, MUTED_TEXT)
        self._draw_text(
            f"Activos A/F: {self.model.vehicle_count} / {self.control_model.vehicle_count}",
            x,
            100,
            self.small_font,
        )
        self._draw_text(
            f"Atendidos A/F: {self.model.completed_count} / {self.control_model.completed_count} · giros {self.model.turns_taken} / {self.control_model.turns_taken}",
            x,
            119,
            self.small_font,
        )
        self._draw_text(
            f"Espera A/F: {self.model.average_wait_time:.1f} / {self.control_model.average_wait_time:.1f} s",
            x,
            138,
            self.small_font,
        )
        self._draw_text(
            f"Viaje A/F: {self.model.average_trip_time:.1f} / {self.control_model.average_trip_time:.1f} s",
            x,
            157,
            self.small_font,
        )
        self._draw_text(
            f"Alertas A: {self.model.alerts_sent}   Cola máx. A/F: {self.model.max_queue_length} / {self.control_model.max_queue_length}",
            x,
            176,
            self.small_font,
        )
        rate_label = "ALTA" if self.model.heavy_traffic_active else "NORMAL"
        self._draw_text(
            f"Demanda: {rate_label}   t={self.model.simulation_time:.1f}s",
            x,
            195,
            self.small_font,
            YELLOW_COLOR if self.model.heavy_traffic_active else GREEN_COLOR,
        )

        pygame.draw.line(
            self.screen,
            (66, 75, 86),
            (x, 209),
            (panel_right, 209),
            1,
        )
        self._draw_text("COLA RECIENTE · A adaptativo / F fijo", x, 215, self.small_font, MUTED_TEXT)
        self._draw_queue_chart(pygame.Rect(x, 231, PANEL_WIDTH - 36, 82))

        pygame.draw.line(
            self.screen,
            (66, 75, 86),
            (x, 320),
            (panel_right, 320),
            1,
        )
        self._draw_text("COOPERACIÓN", x, 329, self.small_font, MUTED_TEXT)
        event_color = (
            YELLOW_COLOR
            if "alerta" in self.model.last_event.lower()
            else MUTED_TEXT
        )
        self._draw_text(self.model.last_event[:35], x, 349, self.tiny_font, event_color)
        self._draw_text(
            "Mensajes dentro del proceso · pulso=alerta",
            x,
            368,
            self.tiny_font,
            (154, 213, 245),
        )

        pygame.draw.line(
            self.screen,
            (66, 75, 86),
            (x, 391),
            (panel_right, 391),
            1,
        )

        heavy_color = (
            (152, 94, 41)
            if self.model.heavy_traffic_active
            else (47, 116, 80)
        )
        heavy_label = (
            "TRÁFICO PESADO ACTIVO"
            if self.model.heavy_traffic_active
            else "INYECTAR TRÁFICO PESADO"
        )
        self._draw_button(self.heavy_button, heavy_label, heavy_color)
        self._draw_button(
            self.pause_button,
            "REANUDAR" if self.paused else "PAUSAR",
            (66, 91, 115),
        )
        self._draw_button(self.reset_button, "REINICIAR", (74, 82, 94))
        self._draw_button(self.speed_down_button, "-", (47, 65, 79))
        self._draw_button(self.speed_up_button, "+", (47, 65, 79))
        speed_label = self.small_font.render(
            f"Velocidad {self.speed_multiplier:g}x",
            True,
            TEXT_COLOR,
        )
        speed_area = pygame.Rect(
            self.speed_down_button.right + 4,
            self.speed_down_button.y,
            self.speed_up_button.x - self.speed_down_button.right - 8,
            self.speed_down_button.height,
        )
        self.screen.blit(speed_label, speed_label.get_rect(center=speed_area.center))
        self._draw_button(
            self.export_button,
            "EXPORTAR MÉTRICAS CSV",
            (43, 91, 126),
        )

    def _draw_control_panel(self) -> None:
        display = self.screen
        self.screen = self.panel_surface
        self.screen.fill(PANEL_BG)
        x = 18
        right = PANEL_WIDTH - 18

        self._draw_text("INDICADORES EN TIEMPO REAL", x, 22, self.section_font, MUTED_TEXT)
        color_by_state = {
            GREEN: GREEN_COLOR,
            YELLOW: YELLOW_COLOR,
            RED: RED_COLOR,
        }
        card_width = 102
        card_gap = 9
        metrics = (
            ("VEHÍCULOS", str(self.model.vehicle_count), "en circulación", (91, 218, 239)),
            ("ESPERA MEDIA", f"{self.model.average_wait_time:.1f} s", "por vehículo", YELLOW_COLOR),
            ("ALERTAS", str(self.model.alerts_sent), "entre agentes", GREEN_COLOR),
        )
        for index, (label, value, note, accent) in enumerate(metrics):
            card = pygame.Rect(x + index * (card_width + card_gap), 39, card_width, 62)
            pygame.draw.rect(self.screen, (25, 39, 54), card, border_radius=7)
            pygame.draw.rect(self.screen, (49, 68, 83), card, width=1, border_radius=7)
            self._draw_text(label, card.x + 9, card.y + 6, self.section_font, MUTED_TEXT)
            self._draw_text(value, card.x + 9, card.y + 20, self.metric_font, TEXT_COLOR)
            self._draw_text(note, card.x + 9, card.y + 46, self.tiny_font, accent)

        self._draw_text("COMPARATIVA DE COLAS", x, 121, self.section_font, MUTED_TEXT)
        self._draw_text("Adaptativo vs. tiempo fijo", x, 138, self.tiny_font, MUTED_TEXT)
        self._draw_queue_chart(pygame.Rect(x, 143, right - x, 104))

        pygame.draw.line(self.screen, (54, 70, 84), (x, 271), (right, 271), 1)
        self._draw_text("INTERSECCIÓN SELECCIONADA", x, 286, self.section_font, MUTED_TEXT)
        central = self.model.signals_by_position.get((1, 1), self.model.signals[0])
        intersection = pygame.Rect(x, 300, right - x, 65)
        pygame.draw.rect(self.screen, (24, 39, 54), intersection, border_radius=8)
        pygame.draw.rect(self.screen, (50, 77, 94), intersection, width=1, border_radius=8)
        number_box = pygame.Rect(x + 9, 310, 45, 45)
        pygame.draw.rect(self.screen, (17, 56, 68), number_box, border_radius=7)
        pygame.draw.rect(self.screen, (34, 166, 191), number_box, width=1, border_radius=7)
        signal_number = central.unique_id or 0
        self._draw_text(f"{signal_number:02d}", number_box.x + 9, number_box.y + 6, self.metric_font, (89, 225, 241))
        self._draw_text(
            f"NODO {signal_number:02d} · CRUCE CENTRAL",
            x + 64,
            308,
            self.section_font,
            TEXT_COLOR,
        )
        phase_name = {
            GREEN: "VERDE",
            YELLOW: "AMARILLO",
            RED: "ROJO",
        }[central.state]
        phase_axis = "TODOS" if central.state == RED else (
            "E-O" if central.green_axis == EW else "N-S"
        )
        phase_badge = pygame.Rect(x + 64, 326, 137, 20)
        pygame.draw.rect(self.screen, (20, 71, 57), phase_badge, border_radius=10)
        pygame.draw.circle(self.screen, color_by_state[central.state], (phase_badge.x + 11, phase_badge.centery), 4)
        self._draw_text(
            f"{phase_axis} {phase_name} · {max(0, math.ceil(central.phase_remaining))} s",
            phase_badge.x + 20,
            phase_badge.y + 3,
            self.tiny_font,
            (196, 237, 222),
        )
        self._draw_text(
            f"Cola E-O {central.queue_counts[EW]} veh.   Cola N-S {central.queue_counts[NS]} veh.",
            x + 64,
            350,
            self.tiny_font,
            MUTED_TEXT,
        )

        pygame.draw.line(self.screen, (54, 70, 84), (x, 380), (right, 380), 1)
        self._draw_text("COOPERACIÓN VECINAL", x, 389, self.section_font, MUTED_TEXT)
        cooperation = pygame.Rect(x, 400, right - x, 58)
        pygame.draw.rect(self.screen, (21, 39, 53), cooperation, border_radius=8)
        pygame.draw.rect(self.screen, (36, 99, 119), cooperation, width=1, border_radius=8)
        pygame.draw.circle(self.screen, (32, 196, 220), (x + 17, 418), 5)
        active_count = len(self.model.active_alerts)
        self._draw_text(
            f"ALERTA DE FLUJO · {active_count} ACTIVAS",
            x + 31,
            409,
            self.section_font,
            (105, 222, 236),
        )
        self._draw_text(self.model.last_event[:43], x + 13, 430, self.tiny_font, TEXT_COLOR)
        self._draw_text(
            "Mensajes simulados dentro del proceso Python",
            x + 13,
            446,
            self.tiny_font,
            MUTED_TEXT,
        )

        pygame.draw.line(self.screen, (54, 70, 84), (x, 466), (right, 466), 1)
        self._draw_text("SIMULACIÓN", x, 477, self.section_font, MUTED_TEXT)
        heavy_color = (152, 94, 41) if self.model.heavy_traffic_active else (27, 126, 89)
        heavy_label = "TRÁFICO PESADO ACTIVO" if self.model.heavy_traffic_active else "INYECCIÓN DE TRÁFICO PESADO"
        self._draw_button(self.heavy_button, heavy_label, heavy_color)
        self._draw_button(self.pause_button, "REANUDAR" if self.paused else "PAUSAR", (41, 65, 83))
        self._draw_button(self.reset_button, "REINICIAR", (35, 52, 69))
        speed_track = pygame.Rect(x, 580, right - x, 28)
        pygame.draw.rect(self.screen, (24, 38, 52), speed_track, border_radius=8)
        pygame.draw.rect(self.screen, (52, 73, 91), speed_track, width=1, border_radius=8)
        self._draw_button(self.speed_down_button, "−", (38, 58, 75))
        self._draw_button(self.speed_up_button, "+", (38, 58, 75))
        speed_area = pygame.Rect(self.speed_down_button.right + 4, 580, self.speed_up_button.x - self.speed_down_button.right - 8, 28)
        speed_label = self.small_font.render(
            f"VELOCIDAD {self.speed_multiplier:g}×",
            True,
            TEXT_COLOR,
        )
        self.screen.blit(speed_label, speed_label.get_rect(center=speed_area.center))
        export_label = self.header_sub_font.render("EXPORTAR DATOS CSV", True, (53, 207, 230))
        self.screen.blit(export_label, (x, 613))

        self.screen = display
        display.blit(self.panel_surface, (CITY_WIDTH, HEADER_HEIGHT))

    def draw(self) -> None:
        self._draw_map()
        self._draw_control_panel()
        self._draw_header()
        self.display.fill(PANEL_BG)
        output = self._fitted_output_rect()
        scale = output.width / DISPLAY_SIZE[0]
        header_height = min(output.height, max(1, round(DISPLAY_HEADER_HEIGHT * scale)))
        map_width = min(output.width, max(1, round(DISPLAY_MAP_SIZE[0] * scale)))
        map_height = output.height - header_height
        panel_width = output.width - map_width

        # Escala cada zona una sola vez para conservar la nitidez al ajustar la ventana.
        self.display.blit(
            pygame.transform.smoothscale(
                self.header_surface,
                (output.width, header_height),
            ),
            (output.x, output.y),
        )
        self.display.blit(
            pygame.transform.smoothscale(self.map_surface, (map_width, map_height)),
            (output.x, output.y + header_height),
        )
        self.display.blit(
            pygame.transform.smoothscale(self.panel_surface, (panel_width, map_height)),
            (output.x + map_width, output.y + header_height),
        )
        pygame.display.flip()

    def handle_events(self) -> None:
        resize_events = {pygame.VIDEORESIZE}
        window_resized = getattr(pygame, "WINDOWRESIZED", None)
        if window_resized is not None:
            resize_events.add(window_resized)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type in resize_events:
                if event.type == pygame.VIDEORESIZE:
                    width, height = event.w, event.h
                else:
                    width, height = event.x, event.y
                if width > 0 and height > 0:
                    self.display = pygame.display.set_mode(
                        (max(640, width), max(480, height)),
                        pygame.RESIZABLE,
                    )

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_model()
                elif event.key == pygame.K_n:
                    self.seed += 1
                    self._reset_model()
                elif event.key == pygame.K_e:
                    self._export_metrics_csv()
                elif event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                    self._set_speed(1)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                    self._set_speed(-1)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                design_pos = self._window_to_design(event.pos)
                if design_pos is None:
                    continue
                if design_pos[0] < DISPLAY_MAP_SIZE[0] or design_pos[1] < DISPLAY_HEADER_HEIGHT:
                    continue
                panel_pos = (
                    int((design_pos[0] - DISPLAY_MAP_SIZE[0]) * PANEL_WIDTH / DISPLAY_PANEL_SIZE[0]),
                    int((design_pos[1] - DISPLAY_HEADER_HEIGHT) * PANEL_HEIGHT / DISPLAY_PANEL_SIZE[1]),
                )
                if self.heavy_button.collidepoint(panel_pos):
                    self.model.inject_heavy_traffic()
                    self.control_model.inject_heavy_traffic()
                elif self.pause_button.collidepoint(panel_pos):
                    self.paused = not self.paused
                elif self.reset_button.collidepoint(panel_pos):
                    self._reset_model()
                elif self.speed_down_button.collidepoint(panel_pos):
                    self._set_speed(-1)
                elif self.speed_up_button.collidepoint(panel_pos):
                    self._set_speed(1)
                elif self.export_button.collidepoint(panel_pos):
                    self._export_metrics_csv()

    def run(self) -> None:
        try:
            while self.running:
                self.handle_events()

                # Limita dt ante pausas o pérdida temporal de foco.
                dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
                if not self.paused:
                    self.model.advance(dt * self.speed_multiplier)
                    self.control_model.advance(dt * self.speed_multiplier)

                self.draw()
        finally:
            pygame.quit()


def main() -> None:
    SimulationWindow().run()


if __name__ == "__main__":
    main()
